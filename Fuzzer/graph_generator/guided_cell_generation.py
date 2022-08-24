import importlib
import torch
import torch.nn as nn
import itertools
from collections import OrderedDict 
import onnx
import tvm
from onnx import shape_inference
import networkx as nx
from onnx import ModelProto
import hashlib
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import glob
from sklearn_extra.cluster import KMedoids
import numpy as np
from scipy import spatial
from tqdm import tqdm
from sklearn.metrics import calinski_harabasz_score

import op_constraint as opc
import op_projection as opp
import random_graph as rg
import utility
import graph_debug as gd
from config import input_shapes, output_shapes, cell_store_location, graph_gen_parameter, generate_cell_num, cluster_center_num

def torch_to_onnx(cell, random_input, file_name):
    torch.onnx.export(cell, random_input, file_name, export_params=False, do_constant_folding=False)

def load_onnx_graph(file_name):
    model = ModelProto()
    with open(file_name, 'rb') as fid:
        content = fid.read()
        model.ParseFromString(content)
    return model


def onnx_to_nx(graph):
    '''
    Turn onnx.onnx_ONNX_REL_1_7_ml_pb2.GraphProto into networkx representation.
    '''
    outid2node_dict = dict()
    G = nx.DiGraph()
    for op_id, op in enumerate(graph.node):
        G.add_node(op.name, op_type=op.op_type)
        for input_node_id in op.input:
            if input_node_id not in outid2node_dict:
                # Do not consider the constant input or the input from out of the cell.
                continue
            else:
                in_node_name = outid2node_dict[input_node_id]
                G.add_edges_from([(in_node_name, op.name)])
        for output_node_id in op.output:
            if output_node_id not in outid2node_dict:
                outid2node_dict[output_node_id] = op.name
    return G

# Code adapted from https://github.com/benedekrozemberczki/graph2vec/blob/master/src/graph2vec.py
# Paper: https://arxiv.org/abs/1707.05005

def graph_reader(graph, name):
    """
    Function to read the graph and features.
        :param graph: The graph object.
        :return features: Features hash table.
        :return name: Name of the graph.
    """
    features = nx.degree(graph)

    features = {k: v for k, v in features}
    return graph, features, name

def feature_extractor(graph, name, rounds=2):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = graph_reader(graph, name)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

def loop_up_vectors(model, graph_name):
    vec = model.dv["g_" + graph_name]
    return vec

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()


def get_graph2vec_model(feature_collection,
                        vector_size=128,
                        window=0,
                        min_count=5,
                        dm=0,
                        sample=0.0001,
                        workers=4,
                        epochs=100,
                        alpha=0.25): 
    
    model = Doc2Vec(feature_collection,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    dm=dm,
                    sample=sample,
                    workers=workers,
                    epochs=epochs,
                    alpha=alpha)
    return model


'''Visualization
from networkx.drawing.nx_pydot import graphviz_layout
#T = nx.balanced_tree(2, 5)

pos = graphviz_layout(G, prog="dot")
nx.draw(G, pos)
'''

def convert_shape_to_str(shape):
    assert len(shape) == 4
    result = [str(i) for i in shape[1:]]
    result = "_".join(result)
    return result

def generate_cells(root_path, cell_num):
    count = 0
    # prepare output path
    for idx in range(len(input_shapes)):
        dir_prefix = convert_shape_to_str(input_shapes[idx]) + "_" + convert_shape_to_str(output_shapes[idx])
        output_path = os.path.join(root_path, dir_prefix)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    while count < cell_num:
        for idx in range(len(input_shapes)):
            recorder = rg.recorder("saved_graph", "saved_graph")
            dir_prefix = convert_shape_to_str(input_shapes[idx]) + "_" + convert_shape_to_str(output_shapes[idx])
            output_path = os.path.join(root_path, dir_prefix)
            input_shape = input_shapes[idx]
            output_shape = output_shapes[idx]
            test_subgraph = rg.get_sub_graph(input_shape, output_shape, recorder=recorder, 
                                            node_num=graph_gen_parameter["node_num"], 
                                            graph_mode=graph_gen_parameter["graph_mode"], p=graph_gen_parameter["p"])
            test_subgraph = test_subgraph.eval()
            count += 1
            file_name = os.path.join(output_path, "{}.pt".format(count))
            torch.save(test_subgraph, file_name)
            file_name = os.path.join(output_path, "{}_recorder.pt".format(count))
            torch.save(recorder, file_name)
            dummy_input = torch.randn(input_shapes[idx])
            file_name = os.path.join(output_path, "{}.onnx".format(count))
            torch.onnx.export(test_subgraph, dummy_input, file_name, export_params=False, do_constant_folding=False)
            if count >= cell_num:
                break

def collect_features(root_path):
    """
    Giving the root path for generated cells,
    transform them into features for graph2vec
    """
    result = list()
    for idx in range(len(input_shapes)):
        dir_prefix = convert_shape_to_str(input_shapes[idx]) + "_" + convert_shape_to_str(output_shapes[idx])
        output_path = os.path.join(root_path, dir_prefix)
        graph_locations = glob.glob(os.path.join(output_path, "*.onnx"))
        for location in graph_locations:
            model = ModelProto()
            with open(location, 'rb') as fid:
                content = fid.read()
                model.ParseFromString(content)
            model_in_nx = onnx_to_nx(model.graph)
            result.append(feature_extractor(model_in_nx, location))
    return result

def get_key_of_center(centers, vector_record):
    location_idx = list()
    not_found_dict = dict()
    for center_idx, center in enumerate(centers):
        find_target = False
        temp_l1_distance = list()
        for idx, item in enumerate(vector_record):
            location = item[0]
            value = item[1]
            temp_l1_distance.append((idx, max(value - center)))
            if np.allclose(value, center, rtol=1e-2, atol=1e-2):
                location_idx.append(location)
                find_target = True
                break
        if find_target != True:
            not_found_dict[center_idx] = temp_l1_distance
            print("Not found corresponding tensor for {}".format(center_idx))
            raise Exception
    return location_idx
        
def cluster_graph_kmedoid(graph2vec_model_path, num_cent_for_cell):
    result = dict()
    model_dict = dict()
    graph2vec_model = torch.load(graph2vec_model_path)
    index_list = list()
    shape_to_graph_list_mapping = dict()
    # collect vectors according to shape
    for key in graph2vec_model.dv.index_to_key:
        index_list.append(key)
        shape = key.split("/")[2]
        if shape not in shape_to_graph_list_mapping:
            shape_to_graph_list_mapping[shape] = list()
        shape_to_graph_list_mapping[shape].append([key, graph2vec_model.dv[key]])
    for shape in shape_to_graph_list_mapping:
        kmedoids = KMedoids(n_clusters=num_cent_for_cell, random_state=0)
        X = [i[1] for i in shape_to_graph_list_mapping[shape]]
        print("Begin cluster")
        kmedoids.fit(X)
        kmedoid_score = calinski_harabasz_score(X, kmedoids.labels_)
        print("Cluster finished")
        print("Inertia for kmeodids: {}".format(kmedoids.inertia_))
        print("CH_index: kmedoid {}".format(kmedoid_score))
        kmedoids_baseline = KMedoids(n_clusters=num_cent_for_cell, max_iter=0)
        kmedoids_baseline.fit(X)
        baseline_score = calinski_harabasz_score(X, kmedoids_baseline.labels_)
        print("Inertia for baseline: {}".format(kmedoids_baseline.inertia_))
        print("CH_index: baseline {}".format(baseline_score))
        location_of_center = get_key_of_center(kmedoids.cluster_centers_, shape_to_graph_list_mapping[shape])
        result[shape] = location_of_center
        model_dict[shape] = kmedoids
    return result, model_dict

    
    
if __name__ == "__main__":
    generate_cells(cell_store_location, generate_cell_num)
    features = collect_features(cell_store_location)
    model = get_graph2vec_model(features)
    vector_mapping = dict()
    graph2vec_location = os.path.join(cell_store_location, "graph2vec_model.pt")
    torch.save(model, graph2vec_location)

    cluster_location, model_dict = cluster_graph_kmedoid(graph2vec_location, cluster_center_num)
    torch.save(cluster_location, os.path.join(cell_store_location, "cluster_location.pt"))
    torch.save(model_dict, os.path.join(cell_store_location, "cluster_model.pt"))
    
    