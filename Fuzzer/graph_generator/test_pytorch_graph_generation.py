import importlib
import torch
import torch.nn as nn
import itertools
from collections import OrderedDict 
import onnx
import tvm
import argparse
from onnx import shape_inference

import op_constraint as opc
import op_projection as opp
import random_graph as rg
import utility
import graph_debug as gd

graph_gen_parameter = {
    "node_num":5, 
    "graph_mode":"ER", 
    "p":0.5
}

input_shapes = [[1, 3, 224, 224], [1, 6, 112, 112], [1, 12, 56, 56], [1, 24, 28, 28]]
output_shapes = [[1, 6, 112, 112], [1, 12, 56, 56], [1, 24, 28, 28], [1, 36, 14, 14]]

if __name__ == "__main__":
    log_file = "pytorch_pressure_test_log.txt"
    parser = argparse.ArgumentParser(description='test TVM using random graph')
    parser.add_argument('--testNum', help="the max test number", type=int, default=2000)
    args = parser.parse_args()
    test_num = args.testNum
    
    gen_idx = 0
    test_idx = 0
    log = open(log_file, "w")
    error_id = 0
    for i in range(test_num):
        if i % 100 == 0:
            log.write("Already test:{}\n".format(i))
        try:
            recorder = rg.recorder("saved_graph_{}".format(error_id), "saved_graph_{}".format(error_id))
            test_subgraphs = list()
            for idx in range(len(input_shapes)):
                input_shape = input_shapes[idx]
                output_shape = output_shapes[idx]
                test_subgraph1 = rg.get_sub_graph(input_shape, output_shape, recorder=recorder, 
                                                node_num=graph_gen_parameter["node_num"], 
                                                graph_mode=graph_gen_parameter["graph_mode"], p=graph_gen_parameter["p"])
                test_subgraph1 = test_subgraph1.eval()
                test_subgraphs.append(test_subgraph1)

            combined_model = torch.nn.Sequential(*test_subgraphs)
            combined_model = combined_model.eval()
            assert combined_model(torch.randn(input_shapes[0])).shape == tuple(output_shapes[-1])
        except Exception as e:
            log.write("===========\n")
            log.write("Error when testing {} model. This is {} error\n".format(i, error_id))
            log.write("Error info: {}\n".format(e))
            error_id += 1
            recorder.save()
            log.flush()
        log.flush()
    log.close()
