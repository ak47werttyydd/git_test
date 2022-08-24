import onnx
import json
import numpy as np
from collections import OrderedDict, defaultdict
from onnx import helper, TensorProto, numpy_helper
import generation_utility as gu

def _parse_attribute(attributes):
    """Parse the ONNX node attribute
    attribute: onnx_model.graph.node[0].attribute"""
    atts = OrderedDict()
    for att in attributes:
        if att.type == onnx.AttributeProto.INT:
            atts[att.name] = att.i
        elif att.type == onnx.AttributeProto.INTS:
            atts[att.name] = att.ints
        elif att.type == onnx.AttributeProto.FLOAT:
            atts[att.name] = att.f
        elif att.type == onnx.AttributeProto.STRING:
            atts[att.name] = att.s
        elif att.type == onnx.AttributeProto.TENSOR:
            atts[att.name] = att.t
        else:
            assert False, "Unsupported Attribute Type: {}".format(att.type)
    return atts


def _parse_dataType(datatype):
    """Parse the dataType
    """
    onnx_dataType_map = dict()
    for i in onnx.TensorProto.DataType.items():
        onnx_dataType_map[i[1]] = i[0]
    return onnx_dataType_map[datatype]

class GraphInfo:
    """Parse the ONNX graph and save the
    corresponding statistics"""
    def __init__(self, onnx_model):
        # Infer the shape first.
        self.graph = onnx.shape_inference.infer_shapes(onnx_model).graph
        self.pure_input = dict() # pure input to the graph
        self.name2shape = dict() # tensor name to tensor shape
        self.name2type = dict() # tensor name to tensor type. Not include weight
        self.weight = dict() # weight of the op in the graph
        self.output = dict() # output shape
        self.op2attribute = defaultdict(list)
        self.parse_graph()
    def _clear(self):
        self.name2shape.clear()
        self.weight.clear()
        self.output.clear()
        self.op2attribute.clear()
    def parse_graph(self, onnx_model=None):
        if onnx_model != None:
            self._clear()
            self.graph = onnx.shape_inference.infer_shapes(onnx_model).graph
        graph = self.graph
        for weight in graph.initializer:
            weight_data = numpy_helper.to_array(weight)
            self.weight[weight.name] = weight_data
            self.name2shape[weight.name] = weight_data.shape
        for input_node in graph.input:
            dims = list()
            for d in input_node.type.tensor_type.shape.dim:
                dims.append(d.dim_value)
            if input_node.name not in self.name2shape:
                self.pure_input[input_node.name] = dims
            self.name2shape[input_node.name] = dims
            self.name2type[input_node.name] = input_node.type.tensor_type.elem_type
        for output_node in graph.output:
            dims = list()
            for d in output_node.type.tensor_type.shape.dim:
                dims.append(d.dim_value)
            self.name2shape[output_node.name] = dims
            self.name2type[output_node.name] = output_node.type.tensor_type.elem_type
        for param_info in graph.value_info:
            dims = list()
            for d in param_info.type.tensor_type.shape.dim:
                dims.append(d.dim_value)
            self.name2shape[param_info.name] = dims
            self.name2type[param_info.name] = param_info.type.tensor_type.elem_type
        for op in graph.node:
            attr_dict = _parse_attribute(op.attribute)
            if op.op_type == "Constant":
                # Constant node may contain necessary tensor for another op's input
                # Only consider single output Constant op
                # Is it OK to record this value in to self.weight?
                assert len(op.output) == 1 
                tensor = list(attr_dict.items())[0][1]
                tensor = numpy_helper.to_array(tensor)
                self.weight[op.output[0]] = tensor
            # TODO: Use hash set to deal with the duplication
            duplication = False
            for attr in self.op2attribute[op.op_type]:
                if attr_dict == attr:
                    duplication = True
                    break
            if duplication == False:
                self.op2attribute[op.op_type].append(attr_dict)
    def get_op_shape(
        self,
        op: onnx.onnx_ml_pb2.NodeProto
    ):
        """Giving an op, return its input,
        weight, output shape."""
        shape_info = OrderedDict()
        shape_info['input'] = OrderedDict()
        shape_info['weight'] = OrderedDict()
        shape_info['output'] = OrderedDict()
        for node_name in op.input:
            assert node_name in self.name2shape
            if node_name not in self.weight:
                shape_info['input'][node_name] = self.name2shape[node_name]
            else:
                shape_info['weight'][node_name] = self.name2shape[node_name]
        for node_name in op.output:
            assert node_name in self.name2shape
            shape_info['output'][node_name] = self.name2shape[node_name]
        return shape_info
    
    def get_tensor_type(self, tensor_name):
        """Find tensor type by given 
        tensor name. Return the type in numpy format"""
        assert isinstance(tensor_name, str)
        if tensor_name in self.weight:
            return self.weight[tensor_name].dtype
        elif tensor_name in self.name2type:
            dtype = self.name2type[tensor_name]
            return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[dtype]
        else:
            raise Exception("Tensor {}, type info not found".format(tensor_name))


        

def get_node_graph(node, graph_info):
    """This function is used to create graph
    which only has one single op. We use this function
    to profile single op.
    
    Parameters
    ----------
    node: onnx.onnx_ml_pb2.NodeProto. 
        The operator.

    graph_info: GraphInfo
        Class GraphInfo is used to hold some statistics
        of the ONNX graph.
    
    return
    ----------
    test_model: 
        The ONNX model which only has one op.
    return_graph_input:
        The input for test model.
    """
    node_param = graph_info.get_op_shape(node)
    node_inputs = list(node.input)
    # Graph input.
    return_graph_input = dict()
    for name in node_param['input']:
        if name in graph_info.weight:
            return_graph_input[name] = graph_info.weight[name]
        else:
            # We only get the shape info, but not the original value
            # So here generate random tensor for test.
            t_type = graph_info.get_tensor_type(name)
            if len(node_param['input'][name]) == 0:
                # single value
                input_tensor_shape = [1]
            else:
                input_tensor_shape = node_param['input'][name]
            if t_type.kind in 'ui':
                # is int
                return_graph_input[name] = np.random.randint(low=0, high=10, size=input_tensor_shape).astype(t_type)
            else:
                return_graph_input[name] = np.random.randn(*input_tensor_shape).astype(t_type)
    # Graph initializer
    graph_init = list()
    for name in node_param['weight']:
        if name in graph_info.weight:
            graph_init.append(numpy_helper.from_array(graph_info.weight[name], name))
        else:
            # generate random tensor if weight is not found.
            t_type = graph_info.get_tensor_type(name)
            weight_tensor = np.random.randn(*node_param['weight'][name]).astype(t_type)
            graph_init.append(numpy_helper.from_array(weight_tensor, name))
    # FIXME: we assume here all tensor is FLOAT. This may lead to problem with int8 model.
    graph_inputs = [
        helper.make_tensor_value_info(
            name, 
            onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[graph_info.get_tensor_type(name)], 
            node_param['input'][name]
            )
        for name in node_param['input']
    ]
    node_outputs = list(node.output)
    graph_outputs = [
        helper.make_tensor_value_info(
            name, 
            onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[graph_info.get_tensor_type(name)], 
            node_param['output'][name])
        for name in node_param['output']
    ]
    node_attr = gu._parse_attribute(node.attribute)
    if node.op_type == "Clip":
        # handle Clip op individually.
        # As in ONNX op version 11, min, max becomes input
        if 'max' in node_attr:
            return_graph_input['max'] = np.array(node_attr['max']).astype(np.float32)
            node_attr.pop('max')
            node_inputs.append('max')
            graph_inputs.append(helper.make_tensor_value_info('max', TensorProto.FLOAT, ()))
        if 'min' in node_attr:
            return_graph_input['min'] = np.array(node_attr['min']).astype(np.float32)
            node_attr.pop('min')
            node_inputs.append('min')
            graph_inputs.append(helper.make_tensor_value_info('min', TensorProto.FLOAT, ()))
    if node.op_type == "Unsqueeze":
        # handle Unsqueeze op individually
        # As in ONNX op version 13, axes becomes input.
        if "axes" in node_attr:
            return_graph_input['axes'] = np.array(node_attr['axes']).astype(np.int64)
            node_attr.pop('axes')
            node_inputs.append('axes')
            graph_inputs.append(helper.make_tensor_value_info('axes', TensorProto.INT64, ()))
    test_node = helper.make_node(
        node.op_type,
        node_inputs,
        node_outputs,
        node.name,
        **node_attr
    )
    single_op_graph = helper.make_graph(
        [test_node], 
        'main', 
        graph_inputs, 
        graph_outputs,
        graph_init
        
    )
    test_model = helper.make_model(single_op_graph, producer_name='single-op-profiler')
    onnx.checker.check_model(test_model) 
    return test_model, return_graph_input
