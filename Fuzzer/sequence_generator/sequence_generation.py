import numpy as np
import pygad
from fast_bleu import BLEU, SelfBLEU
from tqdm import tqdm
from rouge_score import rouge_scorer
import itertools
import torch
import os

from config import gen_sequence_num, gen_sequence_location, test_passes

def generate_random_sequence(sequence_dict, sequence_length, max_opt_level=4):
    all_sequence = list()
    for seq_name in sequence_dict:
        if sequence_dict[seq_name]['opt_level'] > max_opt_level:
            continue
        all_sequence.append(seq_name)
    result_sequence = np.random.choice(all_sequence, sequence_length, replace=True)
    return result_sequence




class GASeqGenerator:
    def __init__(self,
                 metrics="permutation",
                 GA_param=None,
                 init_random_num=10, 
                 sequence_num=500,
                 sequence_length=10,
                 pass_num=28, 
                 alpha=0.7,
                 beta=0.3):
        self.GA_param = GA_param
        self.alpha = alpha
        self.beta = beta
        self.metrics=metrics
        self.init_random_num = init_random_num
        self.sequence_num = sequence_num
        self.sequence_length = sequence_length
        self.pass_num = pass_num
        self.weights = {3: (1/3., 1/3., 1/3.)}
        self.key = 3
        if metrics == "rouge":
            self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.init_random_seeds()
        
    def init_random_seeds(self):
        self.sequence = list()
        all_sequence = [i for i in range(self.pass_num)]
        for i in range(self.init_random_num):
            seq = np.random.choice(all_sequence, self.sequence_length, replace=True)
            self.sequence.append(seq)
        self.sequence_in_token = [[str(i) for i in seq] for seq in self.sequence]
        
    @staticmethod
    def find_ngrams(input_list, n):
        return zip(*[input_list[i:] for i in range(n)])
    
    def get_fitness_function(self):
        # Texygen: A Benchmarking Platform for Text Generation Models 
        # https://dl.acm.org/doi/pdf/10.1145/3209978.3210080?casa_token=JZ9hJqR7lcIAAAAA:TXltUt2BSi7wqYFErVMwLbirvYvvclQXSSj0-JcA09rbXQ6fsP9GR-YBj4Ccyn8QeRUzu5vJKHslW2U
        if self.metrics == "permutation":
            base_set = set()
            for seq in self.sequence:
                base_set = base_set.union(set(GASeqGenerator.find_ngrams(seq, self.key)))
        def fitness_function(solution, solution_idx):
            seq_in_tokens = self.sequence_in_token + [[str(i) for i in solution]]
            solution_token_set = set(solution)
            solution_diverse_score = len(solution_token_set) / len(solution)
            #sol_in_token = [str(i) for i in solution]
            weights = self.weights
            score = 0
            if self.metrics == "blue":
                self_bleu = SelfBLEU(seq_in_tokens, weights)
                score = self_bleu.get_score()[self.key][-1]
                score = 1 - score # diversity
            elif self.metrics == "rouge":
                for i in range(len(seq_in_tokens)-1):
                    score += self.scorer.score(" ".join(seq_in_tokens[i]), 
                                               " ".join(seq_in_tokens[-1]))['rougeL'].fmeasure
                score /= (len(seq_in_tokens)-1)
                score = 1 - score # diversity
            elif self.metrics == "permutation":
                solution_set = set(GASeqGenerator.find_ngrams(solution, self.key))
                new_gram = solution_set.intersection(base_set)
                score = len(new_gram) / len(solution_set)
                score = 1 - score
            return score * self.alpha + solution_diverse_score * self.beta
        return fitness_function
        
    @staticmethod
    def compute_diversity_whole_set(target_set,
                                    method="permutation",
                                    alpha=0.8,
                                    beta=0.2,
                                    weights={4: (1/4., 1/4., 1/4, 1/4)},
                                    pass_num=26):
        # https://aclanthology.org/P19-1177.pdf
        assert(len(weights)) == 1
        key = list(weights.keys())[0]
        length = len(target_set)
        score = 0
        self_diverse_score = 0
        for idx_first in range(len(target_set)):
            self_diverse = len(set(target_set[idx_first])) / len(target_set[idx_first])
            self_diverse_score += self_diverse
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        if method == "permutation":
            permutation_set = set(itertools.permutations([i for i in range(int(pass_num))], int(key)))
            base_set = set()
            for seq in target_set:
                base_set = base_set.union(set(GASeqGenerator.find_ngrams(seq, key)))
            score =  len(base_set) / len(permutation_set)
        else:
            for idx_first in range(len(target_set)):
                for idx_second in range(len(target_set)):
                    if idx_first == idx_second:
                        continue
                    if method == "blue":
                        blue = BLEU([target_set[idx_first]], weights)
                        score += (1 - blue.get_score([target_set[idx_second]])[key][0])
                    elif method == "rouge":
                        score += (1 - scorer.score(" ".join(target_set[idx_first]), " ".join(target_set[idx_second]))['rougeL'].fmeasure)
            score /= (length * (length - 1))
        return alpha * score + beta * self_diverse_score / length
    
    def get_diversity_of_sequnces(self):
        if self.metrics == "permutation":
            input_seq = self.sequence
        else:
            input_seq = self.sequence_in_token
        result = GASeqGenerator.compute_diversity_whole_set(input_seq, self.metrics, self.alpha, self.beta, self.weights, self.pass_num)
        return result
    
    def generate_sequences(self):
        for i in tqdm(range(len(self.sequence), self.sequence_num)):
            fitness_function = self.get_fitness_function()
            if self.GA_param is None:
                ga_instance = pygad.GA(num_generations=100,
                                       num_parents_mating=20,
                                       sol_per_pop=50,
                                       num_genes=self.sequence_length,
                                       mutation_percent_genes=20,
                                       #initial_population=self.sequence,
                                       fitness_func=fitness_function,
                                       random_mutation_min_val=0,
                                       random_mutation_max_val=self.pass_num,
                                       mutation_by_replacement=True,
                                       init_range_low=0,
                                       init_range_high=self.pass_num,
                                       gene_type=int,
                                       gene_space=[i for i in range(self.pass_num)]
                                       )
            else:
                ga_instance = pygad.GA(**self.GA_param)
            ga_instance.run()
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            self.sequence.append(solution)
            self.sequence_in_token.append([str(i) for i in solution])
    
    def get_sequence(self):
        return self.sequence

def get_sequence(idx, sequence_lookup):
    seq = sequence_lookup[idx]
    seq = np.array(seq)
    return test_passes[seq]

if __name__ == "__main__":
    seq_generator = GASeqGenerator(sequence_num=gen_sequence_num)
    seq_generator.generate_sequences()
    random_seq_generator = GASeqGenerator(init_random_num=gen_sequence_num)
    print("Score of GA: {}".format(seq_generator.get_diversity_of_sequnces()))
    print("Score of random: {}".format(random_seq_generator.get_diversity_of_sequnces()))
    if not os.path.exists(gen_sequence_location):
        os.makedirs(gen_sequence_location)
    sequence = seq_generator.sequence
    sorted_seq = sorted(sequence, key=lambda x:[i for i in x])
    torch.save(sorted_seq, os.path.join(gen_sequence_location, "GA_sequence.pt"))
