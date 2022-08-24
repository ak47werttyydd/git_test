import torch.nn as nn
from collections import OrderedDict
import torch
import os
import random
import numpy as np

import op_constraint as opc
from op_constraint import computation_node, activation_node, post_node, layout_node
from utility import get_factors

# activation-computation-normalization
'''
def get_cell(shape):
    """This function samples a cell structure.
    Input shape and output shape of a Cell should be identical.
    
    Activation node: 1
          |
    (layout transform)
          |
    Computation node: 2
          |
    (layout transform)    
          |
    Normalization node: 3
    
    Return: Cell, in format subnode_list, cell_topology, split_tree, shape_info
    """
    subnode_list = list()
    layout_before, layout_after, layout_info = sample_layout_node(shape)
    if layout_info != None:
        layout_pos = random.sample([0, 1, 2, 3], 2)
        layout_pos.sort()
    else:
        layout_pos = None
    cell_graph = _build_cell_graph(shape, layout_info, layout_pos, 4)
    (cell_topology, split_tree, shape_info, op_phase) = cell_graph
    for idx, phase in enumerate(op_phase):
        if phase == -1:
            subnode_list.append(layout_before)
        elif phase == -2:
            subnode_list.append(layout_after)
        elif phase == 1:
            subnode_list.append(sample_activation_node(shape_info[idx + 1][0]))
        elif phase == 2:
            subnode_list.append(sample_computation_node(shape_info[idx + 1][0]))
        elif phase == 3:
            subnode_list.append(sample_post_node(shape_info[idx + 1][0]))
    subnode_list = torch.nn.ModuleList(subnode_list)
    return (subnode_list, cell_topology, split_tree, shape_info)
'''

def get_cell(shape, sample_method=None):
    """This function samples a cell structure and its components(instantiated nodes, topology,shape) instead of assemble a cell
    Input shape and output shape of a Cell should be identical.
    
    Activation node: 1
          |
    (layout transform)
          |
    Computation node: 2
          |
    (layout transform)    
          |
    Normalization node: 3
    
    Return: Cell, in format subnode_list, cell_topology, split_tree, shape_info
    """
    subnode_list = list()
    if sample_method is None:
        #cell_length = random.randint(1, 3)
        cell_length = 3
    else:
        raise Exception("Sample method not defined")
    if cell_length > 1:
        #only sample layout node when the cell length is larger than 1
        #layout node e.g. Splitcat
        layout_before, layout_after, layout_info = sample_layout_node(shape, sample_method)
        # print()
        # print(f'layout sample:\nlayout_before is {layout_before}\nlayout_after is {layout_after}\nlayout_info is{layout_info}')
        #layout_before is SplitWrapper()
        #layout_after is CatWrapper()
        #layout_info is layout_info is[[64, 3, 64, 32], [64, 3, 64, 32]]
    else:
        layout_info = None
    if layout_info is not None:
        # random insert the layerout change node
        layout_pos = random.sample([i for i in range(cell_length+1)], 2) #randomly choose 2 items from a list
        layout_pos.sort()  #in place
    else:
        layout_pos = None
    cell_graph = _build_cell_graph(shape, layout_info, layout_pos, cell_length+1)
    (cell_topology, split_tree, shape_info, op_phase) = cell_graph
    #print('----------------------')
    #print('cell_topology is {0}\nsplit_tree is {1}\nshape_info is {2}\nop_phase is {3}\n'.format(cell_topology, split_tree, shape_info, op_phase))

    #cell_topology is OrderedDict([(1, [0]), (2, [1]), (3, [1]), (4, [1]), (5, [1]), (6, [2]), (7, [3]), (8, [4]), (9, [5]), (10, [6, 7, 8, 9]), (11, [10])])
    #i.e. node0->node1 (node0 is the joint point of multiple previous cells), node1->node2,node3,node4,node5 .....

    #split_tree is OrderedDict([(1, [None]), (2, [0]), (3, [1]), (4, [2]), (5, [3]), (6, [None]), (7, [None]), (8, [None]), (9, [None]), (10, [None, None, None, None]), (11, [None])])
    #only the branch nodes2,3,4,5 has indexes [0],[1],[2],[3] 

    #shape_info is OrderedDict([(0, [[64, 3, 64, 64]]), (1, [[64, 3, 64, 64]]), (2, [[64, 3, 64, 16]]), (3, [[64, 3, 64, 16]]), (4, [[64, 3, 64, 16]]), (5, [[64, 3, 64, 16]]), (6, [[64, 3, 64, 16]]), (7, [[64, 3, 64, 16]]), (8, [[64, 3, 64, 16]]), (9, [[64, 3, 64, 16]]), (10, [[64, 3, 64, 16], [64, 3, 64, 16], [64, 3, 64, 16], [64, 3, 64, 16]]), (11, [[64, 3, 64, 64]])])
    #i.e. Dict(node,list of shapes of inputs)

    #op_phase e.g. [-1, 1, 1, 1, 1, 2, 2, 2, 2, -2, 3]
    # i.e. node0 doesn't represent in op_phase. op_phase[0] is for node1 representing before layout node, node2~5 are op1, node6~9 are op2, node10 is after layout node, node11 is op3

    # The phase
    phase_to_node_list = ["computation", "post", "activation"] #order is the order of the cell topology
    # TODO: consider random sampling?
    #if sample_method is None:
    #    np.random.shuffle(phase_to_node_list)
    
    for idx, phase in enumerate(op_phase):   #every node choose a nn.module layer from the corresponding phases ()
        #idx+1 is the indexes of nodes, starting from 1
        #phase-1 indicates what operators are in nodes
        #every node choose a nn.module layer from the corresponding phases ()
        if phase == -1:
            subnode_list.append(layout_before)   #project a before layout node
        elif phase == -2:
            subnode_list.append(layout_after)    #project an after layout node

        #Only after layout node has multiple inputs. Each other node have just an input, so shape_info[idx+1][0] is the shape of the input
        elif phase_to_node_list[phase-1] == "activation":  #get phase from op_phase from cell_graph
            subnode_list.append(sample_activation_node(shape_info[idx + 1][0], sample_method))
        elif phase_to_node_list[phase-1] == "computation":
            subnode_list.append(sample_computation_node(shape_info[idx + 1][0], sample_method))
        elif phase_to_node_list[phase-1] == "post":
            subnode_list.append(sample_post_node(shape_info[idx + 1][0], sample_method))
    subnode_list = torch.nn.ModuleList(subnode_list) #a ModuleList of instantiated modules
    return (subnode_list, cell_topology, split_tree, shape_info)

    
    
def _build_cell_graph(shape, layout_info, layout_pos, phase_range=4):
    """Helper function for get_cell.
    Build the graph for the cell according to the layout transfer nodes.
    In this function, between the layout transfer nodes must be at least one other node.
    input:
        shape: tuple
            the cell input/output shape
        layout_info: tuple
            the output shape of the first layout transfer node.
            set to None when layout transfer nodes don't exist.
        phase_range: int
            (max idx for subnode) + 1
    Return:
        cell_topology: OrderedDict
            the topology of the cell. Start from index 1.
        split_tree: OrderedDict
            the split tree of the cell
        shape_info: OrderedDict
            the shape of the input of every op
        op_phase: list
            records the phase of every op.
            like:
                -1 : the first layout transfer node.
                -2 : the second layout transfer node.
    """
    cell_topology = OrderedDict()
    split_tree = OrderedDict()
    shape_info = OrderedDict()
    shape_info[0] = [shape]
    before = layout_pos[0] if layout_pos != None else None  #the first place to insert layout node
    after = layout_pos[1] if layout_pos != None else None
    assert (before == None and after == None) or (before >= 0 and before < phase_range)
    assert (before == None and after == None) or (after > 0 and after < phase_range)
    node_index = 1
    split_length = 1
    op_phase = list()
    for phase in range(1, phase_range):
        if before == (phase - 1):
            # add first layout transfomation node
            split_length = len(layout_info) #split to how many nodes
            parent = node_index - 1
            cell_topology[node_index] = [parent]
            split_tree[node_index] =  [None]
            shape_info[node_index] = shape_info[parent]
            parent = node_index  #move on a step
            node_index += 1
            op_phase.append(-1) # -1 indicate the (before) layout transform node
            for i in range(split_length):  #split to how many nodes
                cell_topology[node_index] = [parent]
                split_tree[node_index] = [i]
                shape_info[node_index] = [layout_info[i]]
                node_index += 1 #go to the brother node
                op_phase.append(phase)
        elif after == (phase - 1):
            # add second layout transformation node
            parents = list()
            for i in range(split_length, 0, -1):
                parents.append(node_index-i)
            cell_topology[node_index] = parents
            split_tree[node_index] = [None for i in range(len(parents))]
            node_shape_info = list()
            for t in parents:
                node_shape_info += shape_info[t]
            shape_info[node_index] = node_shape_info
            node_index += 1
            op_phase.append(-2) # -2 indicate the (after) layout transform node
            cell_topology[node_index] = [node_index - 1]
            split_tree[node_index] = [None]
            # FIXME: Here we assert the input shape is identical to the original
            shape_info[node_index] = [shape] 
            split_length = 1
            node_index += 1
            op_phase.append(phase)
        else:
            # deal with normal node.
            for i in range(split_length):
                parent = node_index-split_length
                cell_topology[node_index] = [parent]
                split_tree[node_index] =  [None]
                shape_info[node_index] = shape_info[parent]
                node_index += 1
                op_phase.append(phase)
    if after == (phase_range - 1):
        parents = list()
        for i in range(split_length, 0, -1):
            parents.append(node_index-i)
        cell_topology[node_index] = parents
        split_tree[node_index] = [None for i in range(len(parents))]
        node_shape_info = list()
        for t in parents:
            node_shape_info += shape_info[t]
        shape_info[node_index] = node_shape_info
        op_phase.append(-2) # -2 indicate the (after) layout transform node
    return (cell_topology, split_tree, shape_info, op_phase)
    


def sample_activation_node(shape, method=None):
    if method == None:
        # Default to random method
        global activation_node
        act_num = len(activation_node)
        type_sample_idx = random.randint(0, act_num-1) #choose an activation type
        node = opc.Activation(activation_node[type_sample_idx])
        input_conversion = node.input_conversion(shape, shape) #
        possible_params = node.get_possible_params(node_constraint=input_conversion)
        param_num = len(possible_params) 
        sample_idx = random.randint(0, param_num-1)  #choose a set of params
        act_node = node.get_node(possible_params[sample_idx])
        if activation_node[type_sample_idx] == "LogSoftmax":
            # in TVM "Log softmax requires 2-D input"
            pass
            '''
            input_reshape = opc.ReshapeWrapper((shape[0], -1))
            output_reshape = opc.ReshapeWrapper(shape)
            result_node = nn.Sequential(
                input_reshape, 
                act_node,
                output_reshape
            )
            '''
            result_node = act_node
        else:
            result_node = act_node
        return result_node
        
       
def sample_computation_node(shape, method=None):
    if method == None:
        # default to random method
        global computation_node
        com_num = len(computation_node)
        sample_idx = random.randint(0, com_num-1)
        node = getattr(opc, computation_node[sample_idx])()
        constraint = node.input_conversion(shape, shape)
        params = node.get_possible_params(constraint)
        param_num = len(params)
        sample_idx = random.randint(0, param_num-1)
        return node.get_node(params[sample_idx])
    

def sample_layout_node(shape, method=None):
    if method == None:
        # default to random method
        global layout_node
        layout_num = len(layout_node)   # 1
        sample_idx = random.randint(0, layout_num)
        if sample_idx == layout_num:    # no layout change
            # with NO layout change node.
            layout_before, layout_after, layout_info = None, None, None
        else:
            node = getattr(opc, layout_node[sample_idx])() #instantiate opc.SpliCat
            # print(f'opc.layout_node[sample_idx] is {opc.layout_node[sample_idx]}')
            # print(f'node is {node}')
            # print(f'opc.SplitCat is {opc.SplitCat()}')
            constraint = node.input_conversion(shape, shape)
            params = node.get_possible_params(constraint)
            param_num = len(params)
            sample_idx = random.randint(0, param_num-1)
            layout_before, layout_after = node.get_node(params[sample_idx])
            layout_info = node.get_layout_info(params[sample_idx], shape)
    return layout_before, layout_after, layout_info 

def sample_post_node(shape, method=None):
    if method == None:
        # default to random method
        global post_node
        post_num = len(post_node)
        sample_idx = random.randint(0, post_num-1)
        node = getattr(opc, post_node[sample_idx])()
        constraint = node.input_conversion(shape, shape)
        params = node.get_possible_params(constraint)
        param_num = len(params)
        sample_idx = random.randint(0, param_num-1)
        return node.get_node(params[sample_idx])

class Cell(nn.Module):
    '''get_cell contains the component of a cell.   Cell assemble the these components'''
    """ Cell contains a tree of node.
    """
    def __init__(self,
                 in_degree, 
                 subnode_list: torch.nn.modules.container.ModuleList,
                 cell_topology: OrderedDict,
                 split_tree: OrderedDict,
                 shape_info: OrderedDict = None):
        super(Cell, self).__init__()
        self.in_degree = in_degree
        self.subnode_list = subnode_list
        # cell_topology: node index -> node input
        # start index is 1, as there is a input node to every cell.
        self.cell_topology = cell_topology
        self.split_tree = split_tree
        self.shape_info = shape_info
        self.weights = nn.Parameter(torch.ones(self.in_degree), requires_grad=True)
    
    def forward(self, *input_list):
        assert(len(input_list) == self.in_degree)
        x = 0
        for i in range(self.in_degree):
            x += input_list[i] * self.weights[i]   #weights are 1 initially
        # memory contains the outpout of every subnode
        # cell_topology will guide the op to find its input in memory
        # split_tree will guide the op to find its the part of input this op needed.
        memory = [x]
        node_idx = 1
        #calculate the nodes' input and output in order
        for node in self.subnode_list:  #subnode_list contains instantiated nodes
            sub_node_input = list()

            #search the node_idx node's input 
            for idx, in_vertex in enumerate(self.cell_topology[node_idx]): #in_vertex are the input nodes, idx are input's index
                split_info = self.split_tree[node_idx][idx]
                if split_info != None:
                    sub_node_input.append(memory[in_vertex][split_info])
                else:
                    sub_node_input.append(memory[in_vertex])
            out = node.forward(*sub_node_input) #instantiated module.forward() is equivalent to instantiated module()
            memory.append(out)
            node_idx += 1
        return out  #the end node's out

        


if __name__ == "__main__":
    # Test the function and class
    subnode_list = list()
    relu_node = opc.Activation("ReLU")
    possible_params = relu_node.get_possible_params()
    subnode_list.append(relu_node.get_node(possible_params[0]))
    
    subnode_list.append(opc.SplitWrapper(4, 1))
    
    node_constraint = {
        "H_in": 224, "H_out": 224,
        "W_in": 224, "W_out": 224,
        "C_in": 4, "C_out": 4
    }
    conv2_node = opc.Conv2d()
    possible_params = conv2_node.get_possible_params(node_constraint)
    subnode_list.append(conv2_node.get_node(possible_params[0]))
    subnode_list.append(conv2_node.get_node(possible_params[1]))
    subnode_list.append(opc.CatWrapper(dim=1))
    
    node_constraint = {"num_features": 8}
    batch_node = opc.BatchNorm2d()
    possible_params = batch_node.get_possible_params(node_constraint)
    subnode_list.append(batch_node.get_node(possible_params[0]))
    
    cell_topology = OrderedDict([
        [1, [0]],
        [2, [1]],
        [3, [2]],
        [4, [2]],
        [5, [3, 4]],
        [6, [5]]
    ])
    
    split_tree = OrderedDict([
        [1, [None]],
        [2, [None]],
        [3, [0]],
        [4, [1]],
        [5, [None, None]],
        [6, [None]]
    ])
    
    subnode_list = torch.nn.ModuleList(subnode_list)
    test_cell = Cell(1, subnode_list, cell_topology, split_tree)
    assert test_cell(torch.ones(1, 8, 224, 224)).shape == (1, 8, 224, 224)
    torch.onnx.export(test_cell, torch.ones(1, 8, 224, 224), "test.onnx")
    os.remove("test.onnx")
    
    # test random cell generation
    cell_graph = get_cell((1, 12, 224, 224))
    cell = Cell(1, cell_graph[0], cell_graph[1], cell_graph[2])
    assert cell(torch.ones(1, 12, 224, 224)).shape == (1, 12, 224, 224)
