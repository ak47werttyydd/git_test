from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn
import copy
import tvm

import utility
import random_graph as rg
import op_constraint as opc
import op_projection as opp

class DebugCell(nn.Module):
    """ Same structure as Cell in opp.
    However, this class remove the sum input node.
    By removing the sum input node, multiple inputs are 
    allowed and thus the shape_info becomes OrderedDict of list of shapes.
    
    We need to add a cat node as the output node.
    Because if there exists some disconnected nodes in the graph,
    they will be optimized by TVM. And some input nodes will be removed, causing errors
    in testing.
    """
    
    def __init__(self,
                 in_degree, 
                 subnode_list: torch.nn.modules.container.ModuleList,
                 cell_topology: OrderedDict,
                 split_tree: OrderedDict,
                 shape_info: OrderedDict):
        super(DebugCell, self).__init__()
        self.in_degree = in_degree
        self.subnode_list = subnode_list
        # cell_topology: node index -> node input
        # start index is 1, as there is a input node to every cell.
        self.cell_topology = cell_topology
        self.split_tree = split_tree
        # shape_info here has different meaning with opp.Cell.
        # shape_info should contains the shape of input node.
        self.shape_info = shape_info
        disconnect_nodes = set(cell_topology.keys())
        for i in self.cell_topology:
            for t in self.cell_topology[i]:
                if t in disconnect_nodes:
                    disconnect_nodes.remove(t)
        self.disconnect_nodes = disconnect_nodes
    
    def forward(self, *input_list):
        assert(len(input_list) == self.in_degree)
        memory = list(input_list)
        node_idx = len(input_list) # start from computation node
        for node in self.subnode_list:
            sub_node_input = list()
            for idx, in_vertex in enumerate(self.cell_topology[node_idx]): 
                split_info = self.split_tree[node_idx][idx]
                if split_info != None:
                    sub_node_input.append(memory[in_vertex][split_info])
                else:
                    sub_node_input.append(memory[in_vertex])
            out = node.forward(*sub_node_input)
            memory.append(out)
            node_idx += 1
        return torch.cat(tuple([memory[i].reshape(-1) for i in self.disconnect_nodes]))

def get_sub_dict(ori_dict, key_list):
    """ Return a dict from ori_dict by given key_list
    """
    new_dict = ori_dict.__class__()
    for key in key_list:
        new_dict[key] = copy.deepcopy(ori_dict[key])
    return new_dict

def _get_second_cell(cell_topology, split_tree, shape_info, position):
    cell_topology2 = OrderedDict()
    split_tree2 = OrderedDict()
    shape_info2 = OrderedDict()
    # find input to the second graph
    sec_cell_inputs = defaultdict(list)
    old2parent = defaultdict(list)
    max_idx = max(cell_topology.keys())
    for i in range(position+1, max_idx+1):
        parent_number = 0
        for parent in cell_topology[i]:
            if parent <= position:
                sec_cell_inputs[i].append([parent, parent_number])
            else:
                old2parent[i].append(parent)
            parent_number += 1
    old2new = dict()
    old2new[None] = None
    new_idx = 0
    in_degree = 0
    shape2 = list()
    for i in sec_cell_inputs:
        # Gnereate input node
        for t in sec_cell_inputs[i]:
            parent_number = t[1]
            input_shape = shape_info[i][parent_number]
            shape_info2[new_idx] = [input_shape]
            cell_topology2[new_idx] = [None]
            split_tree2[new_idx] = [None]
            old2parent[i].append(new_idx)
            new_idx += 1
            in_degree += 1
            shape2.append(input_shape)
    input_node_max_idx = new_idx - 1
    for i in range(position+1, max_idx+1):
        old2new[i] = new_idx
        new_idx += 1
    # Deal with the nodes that directly connect to input
    for i in sec_cell_inputs:
        new_idx = old2new[i]
        sorted_parents = sorted(old2parent[i])
        new_parents = list()
        node_shape = list()
        for t in sorted_parents:
            if t <= input_node_max_idx:
                node_shape += shape_info2[t] # input node shape
                new_parents.append(t)
            else:
                node_shape += shape_info[t] # old computation node shape
                new_parents.append(old2new[t])
        cell_topology2[new_idx] = new_parents
        shape_info2[new_idx] = node_shape
        # split tree info is kept in cell_topology
        # So here we set it to None
        split_tree2[new_idx] = [None for t in range(len(old2parent[i]))]
    for i in range(position+1, max_idx+1):
        if i in sec_cell_inputs:
            continue
        new_idx = old2new[i]
        cell_topology2[new_idx] = [old2new[t] for t in cell_topology[i]]
        split_tree2[new_idx] = split_tree[i]
        shape_info2[new_idx] = shape_info[i]
    return in_degree, cell_topology2, split_tree2, shape_info2, shape2
    
def split_cell(cell: opp.Cell, position: int):
    """Split cell by given node position
    shape_info must be included in cell.
    First cell: [0, position]
    Second cell: (position, last_node]
    Notice that this function first split opp.cell
    into debugCell. The reason why we do this is that 
    opp.Cell is somehow inconvenient to debug. 
    
    Input: cell: opp.cell
    
    Return: two Debugcell and their corresponding 
    input shape
    """
    
    assert cell.shape_info != None
    assert position >= 0
    in_degree = cell.in_degree
    subnode_list = cell.subnode_list
    cell_topology = cell.cell_topology
    split_tree = cell.split_tree
    shape_info = cell.shape_info
    input_sum_cell = opc.WeightSum(in_degree) 
    assert position <= len(cell_topology)   
    # build the first cell.
    subnode_list1 = subnode_list[:position]
    subnode_list1.insert(0, input_sum_cell)
    cell_topology1 = OrderedDict()
    split_tree1 = OrderedDict()
    shape_info1 = OrderedDict()
    # init input node
    for i in range(in_degree):
        cell_topology1[i] = [None]
        split_tree1[i] = [None]
        shape_info1[i] = shape_info[0]
    # init sum weight node
    cell_topology1[in_degree] = [i for i in range(in_degree)]
    split_tree1[in_degree] = [None for i in range(in_degree)]
    in_node_shape = list()
    for i in range(in_degree):
        in_node_shape += shape_info[0]
    shape_info1[in_degree] = in_node_shape
    # add shift to intermediate node
    old2new = dict()
    for i in range(0, position+1):
        old2new[i] = i + in_degree
    old2new[None] = None
    for i in range(1, position+1):
        new_idx = old2new[i]
        cell_topology1[new_idx] = [old2new[t] for t in cell_topology[i]]
        split_tree1[new_idx] = split_tree[i]
        #node_shape_info = list()
        #for t in cell_topology[i]:
        #    node_shape_info += shape_info[t]
        shape_info1[new_idx] = shape_info[i]
    cell1 = DebugCell(in_degree, subnode_list1, cell_topology1, split_tree1, shape_info1)
    shape1 = shape_info[0]
    # build the second cell
    subnode_list2 = subnode_list[position:]
    second_cell = _get_second_cell(cell_topology, split_tree, shape_info, position)
    in_degree, cell_topology2, split_tree2, shape_info2, shape2 = second_cell
    cell2 = DebugCell(in_degree, subnode_list2, cell_topology2, split_tree2, shape_info2)
    return (cell1, cell2), (shape1, shape2)

def split_debugCell(cell: DebugCell, position: int):
    # build the first cell.
    assert cell.shape_info != None
    in_degree = cell.in_degree
    subnode_list = cell.subnode_list
    cell_topology = cell.cell_topology
    split_tree = cell.split_tree
    shape_info = cell.shape_info
    
    input_node_num = in_degree
    subnode_list1 = subnode_list[:position]
    cell_topology1 = get_sub_dict(cell_topology, [i for i in range(position+input_node_num)])
    split_tree1 = get_sub_dict(split_tree, [i for i in range(position+input_node_num)])
    shape_info1 = get_sub_dict(shape_info, [i for i in range(position+input_node_num)])
    cell1 = DebugCell(in_degree, subnode_list1, cell_topology1, split_tree1, shape_info1)
    shape1 = list()
    for i in range(in_degree):
        shape1 += shape_info[i]
    
    # build second cell
    subnode_list2 = subnode_list[position:]
    second_cell = _get_second_cell(cell_topology, split_tree, shape_info, position+input_node_num-1)
    in_degree, cell_topology2, split_tree2, shape_info2, shape2 = second_cell
    cell2 = DebugCell(in_degree, subnode_list2, cell_topology2, split_tree2, shape_info2)
    
    return (cell1, cell2), (shape1, shape2)


def test_under_pytorch(model, random_input, exec="normal", opt_level=3, custom_opt_passes=None):
    rtol = 1e-2
    atol = 1e-2
    result = None
    try:
        mod, params = utility.torch2relay(model, random_input)
        if custom_opt_passes is not None:
            with tvm.transform.PassContext(opt_level=0):
                mod, params = tvm.relay.optimize(mod, "llvm", params)
        lib = utility.build_relay(mod, params=params, opt_level=opt_level, custom_opt_passes=custom_opt_passes)
        with torch.no_grad():
            if isinstance(random_input, list) or isinstance(random_input, tuple):
                baseline_outputs = model(*random_input)
                baseline_input = [i.numpy() for i in random_input]
            else:
                baseline_outputs = model(random_input)
                baseline_input = [random_input.numpy()]
            input_name = ["input{}".format(i) for i in range(len(baseline_input))] # See utility.torch2relay
            result = utility.verify_compiled_model(lib, tvm.cpu(0), baseline_input, [baseline_outputs.numpy()], 
                                                   rtol, atol, exec=exec, input_name=input_name, return_tensor=True)
            tvm.testing.assert_allclose(result['tvm'], result['pytorch'], rtol=rtol, atol=atol)
    except Exception as e:
        return (False, e, result)
    else:
        return (True, None, result)

        

def find_bugs_in_subgraph_level(recorder: rg.recorder, test_function, **params):
    error_dict = dict()
    for idx, param in enumerate(recorder.sub_graph_params):
        test_subgraph = rg.TestSubgraph(*param)
        input_shape = param[3]
        random_input = torch.randn(input_shape)
        result = test_function(test_subgraph, random_input, **params)
        if result[0] == False:
            error_dict[idx] = result[1]
    return error_dict

def find_bugs_in_cell_level(cell: opp.Cell, test_function, **params):
    cell_length = len(cell.subnode_list)
    seperate_point = cell_length // 2
    s_cell, shape = split_cell(cell, seperate_point)
    not_seperable = False
    while(not not_seperable):
        error_list = [None, None]
        for idx, i in enumerate(s_cell):
            random_input = utility.get_random_input(shape[idx])
            result = test_function(s_cell[idx], random_input, **params)
            if result[0] == False:
                error_list[idx] = result[1]
        if error_list[0] != None and error_list[1] != None:
            not_seperable = True
            error_result = error_list
            break
        elif error_list[0] != None:
            cell = s_cell[0]
            error_result = error_list[0]
        elif error_list[1] != None:
            cell = s_cell[1]
            error_result = error_list[1]
        else:
            error_result = None
            break
        cell_length = len(cell.subnode_list)
        if cell_length <= 1:
            not_seperable = True
        else:
            seperate_point = cell_length // 2
            s_cell, shape = split_debugCell(cell, seperate_point)
    return (cell, error_result)