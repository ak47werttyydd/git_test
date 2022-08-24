from abc import ABCMeta, abstractmethod
from typing import Sequence
import torch
import numpy as np
import os
import tvm
import time

import pygad
import graph_generator.op_constraint as opc
import graph_generator.guided_cell_generation as gcg
from sequence_generator.pass_analysis_util import get_all_passes_tvm
import sequence_generator.config as sg_config
import graph_generator.config as gg_config
from graph_generator.graph_debug import test_under_pytorch

class GuideTesting(metaclass=ABCMeta):
    def __init__(self):
        pass
    
    @abstractmethod
    def sample_graph(self):
        pass
    
    @abstractmethod
    def sample_sequence(self):
        pass

class GuideTestingGA(GuideTesting):
    def __init__(self, 
                 graph_table_path, 
                 seq_path,
                 alpha=100,
                 beta=0.1,
                 max_batch_size=16,
                 input_shapes=gg_config.input_shapes,
                 output_shapes=gg_config.output_shapes,
                 graph_store_prefix="graph_generator",
                 output_class=1000):
        super(GuideTestingGA, self).__init__()
        self.max_sample_sub_graphs = None
        self.cell_table_path = graph_table_path
        self.seq_path = seq_path
        self.cluster_lookup = torch.load(graph_table_path)
        self.sequence_lookup = torch.load(seq_path)
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.output_class = output_class
        self.graph_store_prefix = graph_store_prefix
        self.alpha = alpha
        self.beta = beta
        self.max_batch_size = max_batch_size
        self.sequence_pool_length = len(self.sequence_lookup)
        self._sort_sub_graphs_table()
        self._sort_sequence()
        self.shape_str_list = list()
        self._process_shape()
        self.recorder = list()
        self.count = 0
    
    def _process_shape(self):
        for idx in range(len(self.input_shapes)):
            in_shape_str = "_".join(str(i) for i in self.input_shapes[idx][1:])
            out_shape_str = "_".join(str(i) for i in self.output_shapes[idx][1:])
            self.shape_str_list.append(in_shape_str + "_" + out_shape_str)

    def _sort_sub_graphs_table(self):
        for key in self.cluster_lookup:
            new_list = []
            for item in self.cluster_lookup[key]:
                location = os.path.join(self.graph_store_prefix, item[2:])
                new_list.append((item, gcg.load_onnx_graph(location).ByteSize()))
            new_list = sorted(new_list, key=lambda x:x[1])
            new_list = [i[0] for i in new_list]
            self.cluster_lookup[key] = new_list
        self.max_sample_sub_graphs = len(new_list)
    
    def _sort_sequence(self):
        pass_opt_level_lookup = get_all_passes_tvm()
        test_passes = sg_config.test_passes
        str_sequence = [test_passes[i] for i in self.sequence_lookup]
        order_sequence_list = list()
        for seq in str_sequence:
            seq_weight = 0
            for opt_pass in seq:
                if opt_pass == "CombineParallelConv2D":
                    opt_pass = "CombineParallelConv2d"
                elif opt_pass == "Inline":
                    opt_pass = "InlineGlobals"
                seq_weight += pass_opt_level_lookup[opt_pass]['opt_level']
            order_sequence_list.append([seq, seq_weight])
        order_sequence_list = sorted(order_sequence_list, key=lambda x:x[1])
        sequence = [i[0] for i in order_sequence_list]
        self.sequence_lookup = sequence
        
    def sample_graph(self, cell_idx_list):
        #
        sub_graphs = list()
        for cell_number, idx in enumerate(cell_idx_list):
            shape = self.shape_str_list[cell_number]
            graph_path = self.cluster_lookup[shape][idx][4:]
            graph_path = os.path.join(self.graph_store_prefix, graph_path)
            graph_path = graph_path.replace("onnx", "pt")
            graph = torch.load(graph_path)
            sub_graphs.append(graph)
        otuput_layer = opc.OutputLayer(self.output_shapes[-1], self.output_class)
        sub_graphs.append(otuput_layer)
        combined_model = torch.nn.Sequential(*sub_graphs)
        combined_model = combined_model.eval()
        return combined_model
    
    def sample_sequence(self, idx):
        seq = self.sequence_lookup[idx]
        temp_list = list()
        for i in seq:
            temp_list.append(getattr(tvm.relay.transform, i)())
        temp_list = tvm.transform.Sequential(temp_list)
        return temp_list
    
    def _test_graph(self, choices):
        batch_size = choices[0]
        input_shape = self.input_shapes[0]
        input_shape[0] = batch_size
        input = torch.randn(input_shape)
        sub_graph_choice = choices[1:4]
        graph = self.sample_graph(sub_graph_choice)
        opt_seq = self.sample_sequence(choices[-1])
        result = test_under_pytorch(graph, input, custom_opt_passes=opt_seq)
        return result
    
    def get_on_generation_function(self):
        def on_generation_function(GA_instance):
            if max(GA_instance.last_generation_fitness) == np.inf:
                return "stop"
        return on_generation_function
    
    def record_error(self, solution):
        self.max_sample_sub_graphs -= 1
        self.sequence_pool_length -= 1
        sequence_idx = solution[0]
        self.sequence_lookup.pop(sequence_idx)
        cell_idx_list = solution[1:-1]
        for cell_number, idx in enumerate(cell_idx_list):
            self.cluster_lookup[self.shape_str_list[cell_number]].pop(idx)
            
    def get_fitness_function(self):
        def fitness_function(solution, solution_idx):
            start_time = time.time()
            result = self._test_graph(solution)
            end_time = time.time()
            self.count += 1
            if result[0] == False:
                # Find bug
                #print(result[1])
                self.recorder.append([solution, result])
                self.record_error(solution)
                score = np.inf
            else:
                output = result[-1]
                max_inf_norm = -1
                for idx in range(len(output['pytorch'])):
                    result_pytorch = output['pytorch'][idx].reshape(-1)
                    result_tvm = output['tvm'][idx].reshape(-1)
                    inf_norm = torch.linalg.norm(torch.tensor(result_pytorch) - torch.tensor(result_tvm), ord=np.inf)
                    if inf_norm > max_inf_norm:
                        max_inf_norm = inf_norm
                score = self.alpha * max_inf_norm + self.beta * (end_time - start_time)
            return score
        return fitness_function
    
    def guide_test(self, limit_time):
        start_time = time.time()
        while(True):
            period_end_time = time.time()
            if (period_end_time - start_time) > limit_time:
                break
            fitness_function = self.get_fitness_function()
            on_generation_function = self.get_on_generation_function()
            ga_instance = pygad.GA(num_generations=100,
                                    num_parents_mating=2,
                                    sol_per_pop=2,
                                    num_genes=5,
                                    mutation_percent_genes=20,
                                    fitness_func=fitness_function,
                                    mutation_by_replacement=True,
                                    on_generation=on_generation_function,
                                    gene_type=int,
                                    gene_space=[[i for i in range(1, self.max_batch_size+1)],
                                                [i for i in range(self.max_sample_sub_graphs)],
                                                [i for i in range(self.max_sample_sub_graphs)],
                                                [i for i in range(self.max_sample_sub_graphs)],
                                                [i for i in range(self.sequence_pool_length)]]
                                    )
            ga_instance.run()

if __name__ == "__main__":
    test_tool = GuideTestingGA(
        graph_table_path = "graph_generator/cell/cluster_location.pt",
        seq_path = "sequence_generator/seq/GA_sequence.pt"
    )
    test_tool.guide_test(gg_config.test_time)
    print(len(test_tool.recorder))
    print(test_tool.count)
    torch.save(test_tool, "guided_test_coverage.pt")