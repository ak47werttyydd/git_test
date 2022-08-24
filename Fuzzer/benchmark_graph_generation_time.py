# This script is used to benchmark the time of different graph generation processes.
import torch
import torch.nn as nn
import itertools
from collections import OrderedDict 
import onnx
import tvm
from onnx import shape_inference
import argparse
import os
import time
import json

import graph_generator.op_constraint as opc
import graph_generator.op_projection as opp
import graph_generator.random_graph as rg
import graph_generator.utility as gutility
import graph_generator.graph_debug as gd


input_shapes = [[1, 3, 224, 224], [1, 6, 112, 112], [1, 12, 56, 56], [1, 24, 28, 28], [1, 12, 7, 7]]
output_shapes = [[1, 6, 112, 112], [1, 12, 56, 56], [1, 24, 28, 28], [1, 36, 14, 14], [1, 12, 7, 7]]

def generate_graph(input_shapes, 
                   output_shapes, 
                   recorder,
                   **graph_parameter):
    test_subgraphs = list()
    for idx in range(len(input_shapes)):
        input_shape = input_shapes[idx]
        output_shape = output_shapes[idx]
        test_subgraph1 = rg.get_sub_graph(input_shape, output_shape, recorder=recorder, **graph_parameter)
        test_subgraph1 = test_subgraph1.eval()
        test_subgraphs.append(test_subgraph1)
    combined_model = torch.nn.Sequential(*test_subgraphs)
    combined_model = combined_model.eval()
    try:
        assert combined_model(torch.randn(input_shapes[0])).shape == tuple(output_shapes[-1])
    except:
        return None
    return combined_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test TVM using random graph')
    parser.add_argument('--onlyRelay', action='store_true', help="Only benchmark relay transformation")
    parser.add_argument('--repeatNum', help="the number of repeated experiments", type=int, default=10)
    parser.add_argument('--cellNodeNum', nargs='+', type=int, help="the number of cell number for benchmarking. Can have multiple values.")
    parser.add_argument('--subGraphNum', nargs="+", type=int, help="the number of sub graph for benchmarking. Can have multiple values")
    parser.add_argument('--graphMode', help="The algorithm for random graph generation", default="ER", type=str)
    parser.add_argument('--opt_level', type=int, help="The optimization level", default=3)
    parser.add_argument('--p', help="The probability for graph generation.", nargs='+', type=float, default=[0.3])
    parser.add_argument('--outpath', help="The path for the output", default="./result.csv", type=str)
    args = parser.parse_args()

    assert max(args.subGraphNum) <= len(input_shapes)
    result = dict()
    graph_gen_parameter = dict()
    graph_gen_parameter["graph_mode"] = args.graphMode
    recorder = rg.recorder(None, None) # dummy recorder
    for graph_num in args.subGraphNum:
        graph_num_id = "sub_graph_num:{}".format(graph_num)
        if graph_num_id not in result:
            result[graph_num_id] = dict()
        for cell_num in args.cellNodeNum:
            graph_gen_parameter["node_num"] = cell_num
            cell_num_id = "cell_num:{}".format(cell_num)
            if cell_num_id not in result[graph_num_id]:
                result[graph_num_id][cell_num_id] = dict()
            for p in args.p:
                p_id = "cell_prob:{}".format(p)
                if p_id not in result[graph_num_id][cell_num_id]:
                    result[graph_num_id][cell_num_id][p_id] = dict()
                graph_gen_parameter["p"] = p
                pytorch_graph_generation_time = 0
                tvm_graph_compile_time = 0
                for i in range(args.repeatNum):
                    # Generate graph for pytorch
                    test_result = False 
                    while (test_result == False):
                        ishape = input_shapes[:graph_num]
                        oshape = output_shapes[:graph_num]
                        pytorch_start_time = time.time()
                        combined_model = generate_graph(
                            ishape,
                            oshape,
                            recorder,
                            **graph_gen_parameter
                        )
                        pytorch_end_time = time.time()
                        test_1 = False if combined_model is None else True
                        random_input = torch.randn(ishape[0])
                        tvm_start_time = time.time()
                        if args.onlyRelay:
                            test_2 = True
                            try:
                                mod, params = gutility.torch2relay(combined_model, random_input)
                                lib = gutility.build_relay(mod, params=params, opt_level=args.opt_level)
                            except:
                                test_2 = False
                        else:
                            test_2 = gd.test_under_pytorch(combined_model, random_input)
                            test_2 = test_2[0]
                        tvm_end_time = time.time()
                        test_result = test_1 and test_2
                        if test_result == True:
                            pytorch_graph_generation_time += pytorch_end_time - pytorch_start_time
                            tvm_graph_compile_time += tvm_end_time - tvm_start_time
                result[graph_num_id][cell_num_id][p_id]['repeated_num'] = args.repeatNum
                result[graph_num_id][cell_num_id][p_id]['ave_pytorch_time'] = pytorch_graph_generation_time / args.repeatNum
                result[graph_num_id][cell_num_id][p_id]['ave_tvm_time'] = tvm_graph_compile_time / args.repeatNum
    with open(args.outpath, "w") as ofile:
        ofile.write(json.dumps(result))




