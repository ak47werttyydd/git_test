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

import op_constraint as opc
import op_projection as opp
import random_graph as rg
import utility
import graph_debug as gd
from config import graph_gen_parameter, input_shapes, output_shapes, test_time  

log_file = "random_graph_test.txt"

def generate_graph(input_shapes, 
                   output_shapes, 
                   recorder, 
                   gen_idx, 
                   log,
                   save_path, 
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
        log.write("===========\n")
        log.write("Generation error. See {}th generation model\n".format(gen_idx))
        torch.save(combined_model, os.path.join(save_path, "{}_generation_error.pt".format(gen_idx)))
        log.flush()
        return None
    return combined_model


# python3 test.py --cellNodeNum 4 --p 0.4 --graphMode ER --logFile cell4_p4_ER_test.txt --graphPath cell4_p4_ER_bug/

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test TVM using random graph')
    #parser.add_argument('--testNum', help="the max test number", type=int, default=1000)
    parser.add_argument('--maxError', help='the max error models to generate', type=int, default=20)
    parser.add_argument('--graphPath', help='the path to save the wrongly generated pytorch graph', type=str, default="./bug/")
    parser.add_argument('--cellNodeNum', help="the node num in one cell", default=4, type=int)
    parser.add_argument('--graphMode', help="The algorithm for random graph generation", default="ER", type=str)
    parser.add_argument('--p', help="The probability for graph generation.", type=float, default=0.4)
    parser.add_argument('--logFile', help="The path for log file", type=str, default="random_graph_test_coverage.txt")
    args = parser.parse_args()
    #test_num = args.testNum
    max_error = args.maxError
    graphPath = args.graphPath
    log_file = args.logFile
    if not os.path.exists(graphPath):
        os.makedirs(graphPath)
    #graph_gen_parameter["node_num"] = args.cellNodeNum
    #graph_gen_parameter["graph_mode"] = args.graphMode
    #graph_gen_parameter["p"] = args.p
    
    gen_idx = 0
    test_idx = 0 
    log = open(log_file, "w")
    recorder_path = os.path.join(args.graphPath, "saved_graph")
    recorder = rg.recorder(recorder_path, recorder_path)
    test_num = 0
    error_num = 0
    start_time = time.time()
    while(True):
        end_time = time.time()
        if end_time - start_time >= test_time:
            break
        combined_model = generate_graph(
            input_shapes,
            output_shapes,
            recorder,
            gen_idx,
            log,
            graphPath,
            **graph_gen_parameter
        )
        if combined_model is None:
            recorder.clear()
            gen_idx += 1
            continue
        random_input = torch.randn(input_shapes[0])
        test_result = gd.test_under_pytorch(combined_model, random_input)
        test_num += 1
        if test_result[0] == True:
            recorder.clear()
            continue
        else:
            error_num += 1
            log.write("===========\n")
            log.write("Error when testing {} model, this is the {}th error\n".format(test_num, test_idx))
            log.write("Please check {}th test_error_model\n".format(test_idx))
            log.write("Error info: {}\n".format(test_result[1]))
            torch.save(recorder, os.path.join(graphPath, "{}_test_error_recorder.pt".format(test_idx)))
            torch.save(combined_model, os.path.join(graphPath, "{}_test_error_model.pt".format(test_idx)))
            test_idx += 1
            recorder.clear()
            log.flush()
        if test_idx >= max_error:
            log.write("Max error model generation finished, exit\n")
            log.write("===\n")
            log.write("Test num: {} for now".format(test_num))
            log.flush()
            break
    log.write("Test num: {}\n".format(test_num))
    log.write("Error num: {}\n".format(error_num))
    log.write("TEST FINISHED\n")
    log.close()