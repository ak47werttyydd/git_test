import constraint
import math
import torch
import torch.nn as nn
import itertools
from functools import reduce
from abc import ABCMeta, abstractmethod
import random
from collections import OrderedDict

import utility
from utility import get_combine_list, get_factors

# FIXME
#"Bilinear"
#"ConvTranspose2d" has bug in TVM now.
#"SeparableConv2d"
computation_node = [
    "Conv2d", "ConvTranspose2d", 
    "SeparableConv2d", "Linear"    #"SeparableConv2d" is depthwise Conv2d+ pointwiseConv
    ]

activation_node = ['ReLU', 'ReLU6', 'Sigmoid', 
                   'LogSigmoid', 'Tanh', 'LeakyReLU', 
                   'PReLU', 'Softmin', 'Softmax'
                   ]

# remove 'LogSoftmax'

post_node = [
    "BatchNorm2d",
    "GroupNorm",
    "Dropout"
    ]

layout_node = [
    "SplitCat"
    ]

class Subnode(metaclass=ABCMeta):   #abstractSplit class

    @abstractmethod
    def get_possible_params(self, node_constraint: dict):
        pass

    @abstractmethod
    def get_node(self, param: dict):
        pass
    
#    @abstractmethod
#    def get_split_info(self):
#        pass

class Conv2d(Subnode):
    def __init__(self):
        super(Conv2d, self).__init__()
        self.solver = constraint.Problem()
        self.constraint_property = [
            "H_in", "H_out", "W_in", "W_out", "C_in", "C_out"]
        self.in_channels = None
        self.out_channels = None
        self.default_variable_domain = {
            "padding0": [i for i in range(0, 4)],
            "padding1": [i for i in range(0, 4)],
            "dilation0": [i for i in range(1, 4)],
            "dilation1": [i for i in range(1, 4)],
            "kernel_size0": [i for i in range(1, 8)],
            "kernel_size1": [i for i in range(1, 8)],
            "stride0": [i for i in range(1, 3)],
            "stride1": [i for i in range(1, 3)],
            "groups": [1, 2, 3],
            "bias": [True, False]
        }

    def _H_constraint(self, H_in, H_out):
        """Height constraint for conv2d
        """
        return (lambda padding0, dilation0, kernel_size0, stride0:
                H_out == math.floor(
                    ((H_in + 2*padding0 - dilation0*(kernel_size0 - 1) - 1) / stride0) + 1)
                )

    def _W_constraint(self, W_in, W_out):
        """Weight constraint for conv2d
        """
        return (lambda padding1, dilation1, kernel_size1, stride1:
                W_out == math.floor(
                    ((W_in + 2*padding1 - dilation1 * (kernel_size1 - 1) - 1) / stride1) + 1)
                )

    def _C_constraint(self, C_in, C_out):
        """Channel constraint for conv2d
        """
        return (lambda groups: C_in % groups == 0 and C_out % groups == 0)

    def _param_conversion(self, results: list):
        converted_result = list()
        for result in results:
            stride = (result['stride0'], result['stride1'])
            dilation = (result['dilation0'], result['dilation1'])
            padding = (result['padding0'], result['padding1'])
            kernel_size = (result['kernel_size0'], result['kernel_size1'])
            groups = (result['groups'])
            converted_result.append(
                {
                    "stride": stride,
                    "dilation": dilation,
                    "padding": padding,
                    "kernel_size": kernel_size,
                    "groups": groups,
                    "in_channels": self.in_channels,
                    "out_channels": self.out_channels,
                    "bias": result["bias"]
                }
            )
        return converted_result

    def input_conversion(self, input_shape, output_shape):
        # asset input shape to be NCHW.
        node_constraint = dict()
        node_constraint["C_in"] = input_shape[1]
        node_constraint["H_in"] = input_shape[2]
        node_constraint["W_in"] = input_shape[3]
        node_constraint["C_out"] = output_shape[1]
        node_constraint["H_out"] = output_shape[2]
        node_constraint["W_out"] = output_shape[3]
        return node_constraint
    
    def get_possible_params(self, node_constraint, param_domain=None):
        """Given input, output constraint, 
        return possible parameter combination"""
        for cons_prop in self.constraint_property:
            assert cons_prop in node_constraint
        self.solver.reset()
        H_in, H_out = node_constraint["H_in"], node_constraint["H_out"]
        W_in, W_out = node_constraint["W_in"], node_constraint["W_out"]
        C_in, C_out = node_constraint["C_in"], node_constraint["C_out"]
        if C_out % C_in == 0:
            if C_in not in self.default_variable_domain["groups"]:
                # Possibility of depthwise conv
                self.default_variable_domain["groups"].append(C_in)

        for domain in self.default_variable_domain:
            if param_domain != None and domain in param_domain:
                self.solver.addVariable(domain, param_domain[domain])
            else:
                self.solver.addVariable(
                    domain, self.default_variable_domain[domain])
        self.in_channels = C_in
        self.out_channels = C_out
        self.solver.addConstraint(self._H_constraint(H_in, H_out),
                                  ("padding0", "dilation0", "kernel_size0", "stride0"))
        self.solver.addConstraint(self._W_constraint(W_in, W_out),
                                  ("padding1", "dilation1", "kernel_size1", "stride1"))
        self.solver.addConstraint(self._C_constraint(C_in, C_out),
                                  ("groups", ))
        result = self.solver.getSolutions()
        return self._param_conversion(result)

    def get_node(self, param):
        return nn.Conv2d(**param)

    def get_split_info(self):
        raise NotImplementedError

# Please refer to https://github.com/apache/tvm/blob/4344540ad4206733dd136678180fbf7e3dd616c3/python/tvm/relay/op/strategy/generic.py#L471
# For parameters that TVM doesn't support.

class ConvTranspose2d(Subnode):
    def __init__(self):
        super(ConvTranspose2d, self).__init__()
        self.solver = constraint.Problem()
        self.constraint_property = [
            "H_in", "H_out", "W_in", "W_out", "C_in", "C_out"]
        self.in_channels = None
        self.out_channels = None
        self.default_variable_domain = {
            "padding0": [i for i in range(0, 4)],
            "padding1": [i for i in range(0, 4)],
            "output_padding0": [i for i in range(0, 2)],
            "output_padding1": [i for i in range(0, 2)],
            "dilation0": [1],
            "dilation1": [1],
            "kernel_size0": [i for i in range(1, 8)],
            "kernel_size1": [i for i in range(1, 8)],
            "stride0": [i for i in range(1, 3)],
            "stride1": [i for i in range(1, 3)],
            "groups": [1],
            "bias": [True, False],
        }

    def _H_constraint(self, H_in, H_out):
        """Height constraint for ConvTranspose2d
        """
        return (lambda padding0, output_padding0, dilation0, kernel_size0, stride0:
                H_out == (H_in - 1) * stride0 - 2 * padding0 +
                dilation0 * (kernel_size0 - 1) + output_padding0 + 1
                )

    def _W_constraint(self, W_in, W_out):
        """Weight constraint for ConvTranspose2d
        """
        return (lambda padding1, output_padding1, dilation1, kernel_size1, stride1:
                W_out == (W_in - 1) * stride1 - 2 * padding1 +
                dilation1 * (kernel_size1 - 1) + output_padding1 + 1
                )

    def _C_constraint(self, C_in, C_out):
        """Channel constraint for conv2d
        """
        return (lambda groups: C_in % groups == 0 and C_out % groups == 0)

    def _output_padding_constraint(self):
        """output padding must be smaller than either stride or dilation
        """
        return (lambda output_padding0, output_padding1, stride0, stride1, dilation0, dilation1:
                (
                    (output_padding0 < stride0 or output_padding0 < dilation0) and
                    (output_padding1 < stride1 or output_padding1 < dilation1)
                    ))

    def _param_conversion(self, results: list):
        converted_result = list()
        for result in results:
            stride = (result['stride0'], result['stride1'])
            dilation = (result['dilation0'], result['dilation1'])
            padding = (result['padding0'], result['padding1'])
            output_padding = (result['output_padding0'],
                              result['output_padding1'])
            kernel_size = (result['kernel_size0'], result['kernel_size1'])
            groups = (result['groups'])
            converted_result.append(
                {
                    "stride": stride,
                    "dilation": dilation,
                    "padding": padding,
                    "output_padding": output_padding,
                    "kernel_size": kernel_size,
                    "groups": groups,
                    "in_channels": self.in_channels,
                    "out_channels": self.out_channels,
                    "bias": result["bias"]
                }
            )
        return converted_result
    
    def input_conversion(self, input_shape, output_shape):
        # asset input shape to be NCHW.
        node_constraint = dict()
        node_constraint["C_in"] = input_shape[1]
        node_constraint["H_in"] = input_shape[2]
        node_constraint["W_in"] = input_shape[3]
        node_constraint["C_out"] = output_shape[1]
        node_constraint["H_out"] = output_shape[2]
        node_constraint["W_out"] = output_shape[3]
        return node_constraint
    
    def get_possible_params(self, node_constraint, param_domain=None):
        """Given input, output constraint, 
        return possible parameter combination"""
        for cons_prop in self.constraint_property:
            assert cons_prop in node_constraint
        self.solver.reset()
        H_in, H_out = node_constraint["H_in"], node_constraint["H_out"]
        W_in, W_out = node_constraint["W_in"], node_constraint["W_out"]
        C_in, C_out = node_constraint["C_in"], node_constraint["C_out"]
        if C_out % C_in == 0:
            if C_in not in self.default_variable_domain["groups"]:
                # Possibility of depthwise conv
                # TVM doesn't support for groups > 1 for now.
                # self.default_variable_domain["groups"].append(C_in)
                pass

        for domain in self.default_variable_domain:
            if param_domain != None and domain in param_domain:
                self.solver.addVariable(domain, param_domain[domain])
            else:
                self.solver.addVariable(
                    domain, self.default_variable_domain[domain])
        self.in_channels = C_in
        self.out_channels = C_out
        self.solver.addConstraint(self._H_constraint(H_in, H_out),
                                  ("padding0", "output_padding0",
                                   "dilation0", "kernel_size0", "stride0")
                                  )
        self.solver.addConstraint(self._W_constraint(W_in, W_out),
                                  ("padding1", "output_padding1", 
                                   "dilation1", "kernel_size1", "stride1"))
        self.solver.addConstraint(self._C_constraint(C_in, C_out),
                                  ("groups", ))
        self.solver.addConstraint(self._output_padding_constraint(),
                                  ("output_padding0", "output_padding1", "stride0", 
                                   "stride1", "dilation0", "dilation1"))
        result = self.solver.getSolutions()
        return self._param_conversion(result)

    def get_node(self, param):
        return nn.ConvTranspose2d(**param)

    def get_split_info(self):
        raise NotImplementedError

class SeparableConv2d(Subnode):
    def __init__(self):
        super(SeparableConv2d, self).__init__()
        self.solver = constraint.Problem()
        self.constraint_property = [
            "H_in", "H_out", "W_in", "W_out", "C_in", "C_out"
            ]
        self.in_channels = None
        self.out_channels = None
        self.default_variable_domain = {
            "padding0": [i for i in range(0, 4)],
            "padding1": [i for i in range(0, 4)],
            "dilation0": [i for i in range(1, 4)],
            "dilation1": [i for i in range(1, 4)],
            "kernel_size0": [3, 5],
            "kernel_size1": [3, 5],
            "stride0": [i for i in range(1, 3)],
            "stride1": [i for i in range(1, 3)],
            "groups": None,
            "bias": [True, False]
        }
    def _H_constraint(self, H_in, H_out):
        """Height constraint for conv2d
        """
        return (lambda padding0, dilation0, kernel_size0, stride0:
                H_out == math.floor(
                    ((H_in + 2*padding0 - dilation0*(kernel_size0 - 1) - 1) / stride0) + 1)
                )

    def _W_constraint(self, W_in, W_out):
        """Weight constraint for conv2d
        """
        return (lambda padding1, dilation1, kernel_size1, stride1:
                W_out == math.floor(
                    ((W_in + 2*padding1 - dilation1 * (kernel_size1 - 1) - 1) / stride1) + 1)
                )

    def _param_conversion(self, results: list):
        converted_result = list()
        for result in results:
            stride = (result['stride0'], result['stride1'])
            dilation = (result['dilation0'], result['dilation1'])
            padding = (result['padding0'], result['padding1'])
            kernel_size = (result['kernel_size0'], result['kernel_size1'])
            groups = (result['groups'])
            converted_result.append(
                {
                    "stride": stride,
                    "dilation": dilation,
                    "padding": padding,
                    "kernel_size": kernel_size,
                    "groups": groups,
                    "in_channels": self.in_channels,
                    "out_channels": self.out_channels,
                    "bias": result["bias"]
                }
            )
        return converted_result

    def input_conversion(self, input_shape, output_shape):
        # asset input shape to be NCHW.
        node_constraint = dict()
        node_constraint["C_in"] = input_shape[1]
        node_constraint["H_in"] = input_shape[2]
        node_constraint["W_in"] = input_shape[3]
        node_constraint["C_out"] = output_shape[1]
        node_constraint["H_out"] = output_shape[2]
        node_constraint["W_out"] = output_shape[3]
        return node_constraint
    
    def get_possible_params(self, node_constraint, param_domain=None):
        """Given input, output constraint, 
        return possible parameter combination"""
        for cons_prop in self.constraint_property:
            assert cons_prop in node_constraint
        self.solver.reset()
        H_in, H_out = node_constraint["H_in"], node_constraint["H_out"]
        W_in, W_out = node_constraint["W_in"], node_constraint["W_out"]
        C_in, C_out = node_constraint["C_in"], node_constraint["C_out"]
        self.default_variable_domain["groups"] = [C_in]

        for domain in self.default_variable_domain:
            if param_domain != None and domain in param_domain:
                self.solver.addVariable(domain, param_domain[domain])
            else:
                self.solver.addVariable(
                    domain, self.default_variable_domain[domain])
        self.in_channels = C_in
        self.out_channels = C_out
        self.solver.addConstraint(self._H_constraint(H_in, H_out),
                                  ("padding0", "dilation0", "kernel_size0", "stride0"))
        self.solver.addConstraint(self._W_constraint(W_in, W_out),
                                  ("padding1", "dilation1", "kernel_size1", "stride1"))
        result = self.solver.getSolutions()
        return self._param_conversion(result)

    def get_node(self, param):
        conv_param = param.copy()
        conv_param["out_channels"] = conv_param["in_channels"]
        conv_param["groups"] = conv_param["in_channels"]
        point_param = {
            "in_channels": conv_param["in_channels"],
            "out_channels": conv_param["out_channels"],
            "bias": conv_param["bias"],
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
        }
        return nn.Sequential(OrderedDict([
                    ("depthwiseConv", nn.Conv2d(**conv_param)),
                    ("pointwiseConv", nn.Conv2d(**point_param))
                ]))

    def get_split_info(self):
        raise NotImplementedError
        
class Linear(Subnode):
    def __init__(self):
        super(Linear, self).__init__()
        #self.solver = constraint.Problem()
        self.constraint_property = ["in_features", "out_features"]
        self.default_variable_domain = None

    def input_conversion(self, input_shape, output_shape):
        # Input: (N, *, H_{in}) H_in is the in_features
        # 
        node_constraint = dict()
        node_constraint["in_features"] = input_shape[-1]
        node_constraint["out_features"] = output_shape[-1]
        return node_constraint
    
    def get_possible_params(self, node_constraint, param_domain=None):
        for cons_prop in self.constraint_property:
            assert cons_prop in node_constraint
        if param_domain != None:
            raise NotImplementedError
        result = list()
        for bias in [True, False]:
            result.append({
                "in_features": node_constraint["in_features"],
                "out_features": node_constraint["out_features"],
                "bias": bias
            })
        return result

    def get_node(self, param):
        return nn.Linear(**param)

    def get_split_info(self):
        raise NotImplementedError

class Bilinear(Subnode):
    """As all computation nodes here have only one input.
    We will do the split first. 
    """
    def __init__(self):
        super(Bilinear, self).__init__()
        self.constraint_property = ["in_features", "out_features"]
        self.default_variable_domain = None

    def input_conversion(self, input_shape, output_shape):
        # Input: (N, *, H_{in}) H_in is the in_features
        # 
        node_constraint = dict()
        node_constraint["in_features"] = input_shape[-1]
        node_constraint["out_features"] = output_shape[-1]
        return node_constraint
    
    def get_possible_params(self, node_constraint, param_domain=None):
        for cons_prop in self.constraint_property:
            assert cons_prop in node_constraint
        if param_domain != None:
            raise NotImplementedError
        result = list()
        feat1 = node_constraint["in_features"] // 2
        feat2 = node_constraint["in_features"] - feat1
        node_constraint["in1_features"] = max(feat1, feat2)
        node_constraint["in2_features"] = min(feat1, feat2)
        for bias in [True, False]:
            result.append({
                "in1_features": node_constraint["in1_features"],
                "in2_features": node_constraint["in2_features"],
                "out_features": node_constraint["out_features"],
                "bias": bias
            })
        return result  
    
    def get_node(self, param):
        split_chunk = max(param["in1_features"], param["in2_features"])
        class Comb_Bilinear(nn.Module):
            def __init__(self):
                super(Comb_Bilinear, self).__init__()
                self.split_node = SplitWrapper(split_chunk, -1)
                self.Bilinear = nn.Bilinear(**param)  
            def forward(self, x):
                x1, x2 = self.split_node(x)
                x1 = x1.contiguous()
                x2 = x2.contiguous()
                return self.Bilinear(x1, x2) 
        return Comb_Bilinear()
    
    def get_split_info(self):
        raise NotImplementedError

# Does't allow inplace operation
"""
 UserWarning: Output 1 of SplitBackward is a view and is being modified inplace. 
 This view is an output of a function that returns multiple views. 
 Inplace operators on such views are being deprecated and will be forbidden 
 starting from version 1.8. Consider using `unsafe_` version of the function 
 that produced this view or don't modify this view inplace. 
 (Triggered internally at  ../torch/csrc/autograd/variable.cpp:547.)
"""

"""
TVM Error:
  Check failed: (param->axis == -1 || param->axis == static_cast<int32_t>(inputs[0].ndim()) - 1) 
  is false: log_softmax currently only works on last dimension
"""

act_propoerty_domain = {
    "ReLU": {"inplace": [False]}, 
    "ReLU6": {"inplace": [False]},
    "Sigmoid": {},
    "LogSigmoid": {},
    "Tanh": {},
    "LeakyReLU": {"negative_slope": [i * 0.01 for i in range(1, 10)],
                  "inplace": [False]},
    "PReLU": {
        "num_parameters": [1],
        "init": [i * 0.1 for i in range(1, 10)]},
    "Softmin": {
        "dim": [-1]},
    "Softmax": {
        "dim": [-1]},
    "LogSoftmax": {
        "dim": [-1]}
}

#act_dim_as_property = set(["Softmin", "Softmax"])
act_dim_as_property = set([])

class Activation(Subnode):
    """Activation nodes don't change shape.
    And there is usually no constraint for the op.
    So we wrap them in one class"""
    def __init__(self, act_type: str):
        super(Activation, self).__init__()
        self.constraint_property = None
        self.act_type = act_type
        self.default_variable_domain = act_propoerty_domain[act_type]
    
    def input_conversion(self, input_shape, output_shape):
        assert len(input_shape) == len(output_shape)
        node_constraint = dict()
        node_constraint["shape"] = input_shape
        node_constraint["dim"] = len(input_shape)
        return node_constraint

    def get_possible_params(self, node_constraint=None, param_domain=None):
        variable_domain =  self.default_variable_domain
        if param_domain != None:
            for param in param_domain:
                if param in variable_domain:
                    variable_domain[param] = param_domain[param] 
        if node_constraint != None:
            if self.act_type in act_dim_as_property:
                # for activation node, the dim can be used
                # as potential parameter
                assert "dim" in node_constraint
                dim = node_constraint["dim"]
                variable_domain["dim"] = [i for i in range(dim)]
        if len(variable_domain) == 0:
            # No parameter needed for the op.
            return [None]
        results = get_combine_list(variable_domain)
        return results
    
    def get_node(self, param):
        if param == None:
            # no parameters
            param = dict()
        act_func = getattr(nn, self.act_type) # call activation layers in torch.nn
        return act_func(**param)

    def get_split_info(self):
        raise NotImplementedError


class BatchNorm2d(Subnode):
    def __init__(self):
        super(BatchNorm2d, self).__init__()
        # the num_features are the input property of BatchNorm2d
        # There is in fact no constraint needed to be soleved.
        self.constraint_property = ["num_features"]
        self.default_variable_domain = {
            "eps": [10**(-i) for i in range(1, 6)],
            "momentum": [0.1 * i for i in range(1, 6)],
            "affine": [True, False], # False is not supported.
            "track_running_stats": [True]
        }
    
    def input_conversion(self, input_shape, output_shape):
        # convert input to node constraint
        # Assert shape: NCHW
        node_constraint = dict()
        node_constraint["num_features"] = input_shape[1]
        return node_constraint
    
    def get_possible_params(self, node_constraint=None, param_domain=None):
        variable_domain = self.default_variable_domain
        # user-specific parameter domain
        if param_domain != None:
            for param in param_domain:
                if param in variable_domain:
                    variable_domain[param] = param_domain[param]        
        for prop in self.constraint_property:
            assert prop in node_constraint
        if not isinstance(node_constraint["num_features"], list):
            num_features = [node_constraint["num_features"]]
        else:
            num_features = node_constraint["num_features"]
        variable_domain["num_features"] = num_features
        results = get_combine_list(variable_domain)
        return results
    
    def get_node(self, param):
        return nn.BatchNorm2d(**param)

    def get_split_info(self):
        raise NotImplementedError
    
class GroupNorm(Subnode):
    def __init__(self):
        super(GroupNorm, self).__init__()
        # the num_features are the input property of BatchNorm2d
        # There is in fact no constraint needed to be soleved.
        self.constraint_property = ["num_channels"]
        self.default_variable_domain = {
            "eps": [10**(-i) for i in range(1, 6)],
            "num_groups": [], # We use factors to compute this
            "affine": [True]
        }

    def input_conversion(self, input_shape, output_shape):
        # convert input to node constraint
        # Assert shape: NCHW
        node_constraint = dict()
        node_constraint["num_channels"] = input_shape[1]
        return node_constraint
    
    def get_possible_params(self, node_constraint=None, param_domain=None):
        variable_domain = self.default_variable_domain
        # user-specific parameter domain
        if param_domain != None:
            for param in param_domain:
                if param in variable_domain:
                    variable_domain[param] = param_domain[param]
        for prop in self.constraint_property:
            assert prop in node_constraint
        if not isinstance(node_constraint["num_channels"], list):
            num_channels = [node_constraint["num_channels"]]
        else:
            num_channels = node_constraint["num_channels"]
        variable_domain["num_channels"] = num_channels
        # There is a condition that num_channels % num_groups == 0
        num_groups = list(get_factors(node_constraint["num_channels"]))
        random.shuffle(num_groups)
        num_groups = num_groups[:6] # FIXME Hard-coded. Only select part of the groups.
        variable_domain["num_groups"] = num_groups
        results = get_combine_list(variable_domain)
        return results
    
    def get_node(self, param):
        return nn.GroupNorm(**param)

    def get_split_info(self):
        raise NotImplementedError
        
class Dropout(Subnode):
    def __init__(self):
        super(Dropout, self).__init__()
        self.constraint_property = None
        self.default_variable_domain = {
            "p": [i*0.1 for i in range(1, 10)],
            "inplace": [False]
        }
    
    def input_conversion(self, input_shape, output_shape):
        # No constraint needed for Dropout node
        return None
    
    def get_possible_params(self, node_constraint=None, param_domain=None):
        variable_domain = self.default_variable_domain
        # user-specific parameter domain
        if param_domain != None:
            for param in param_domain:
                if param in variable_domain:
                    variable_domain[param] = param_domain[param]    
        results = get_combine_list(variable_domain)
        return results  
    
    def get_node(self, param):
        return nn.Dropout(**param)

    def get_split_info(self):
        raise NotImplementedError
                
class CompSubnode(Subnode):
    """Computation sub node
    """

    def __init__(self, node_type):
        super(CompSubnode, self).__init__()
        self.node_type = node_type

class NormSubnode(Subnode):
    """Normalization sub node
    """

    def __init__(self, node_type):
        super(NormSubnode, self).__init__()

class ActSubnode(Subnode):
    """Activation sub node
    """

    def __init__(self, node_type):
        super(ActSubnode, self).__init__()

class TransfSubnode(Subnode):
    """Layer transformation subnode
    """

    def __init__(self, node_type):
        super(TransfSubnode, self).__init__()

class SplitWrapper(nn.Module):
    def __init__(self, split_size_or_sections, dim):
        super(SplitWrapper, self).__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim
    
    def forward(self, x):
        return torch.split(x, self.split_size_or_sections, self.dim)

class CatWrapper(nn.Module):
    def __init__(self, dim):
        super(CatWrapper, self).__init__()
        self.dim = dim
    
    def forward(self, *x):
        return torch.cat(x, self.dim)

class SplitCat(Subnode):
    def __init__(self):
        super(SplitCat, self).__init__()
        self.constraint_property = ["input_shape"]
        self.default_variable_domain = {
            "split_size_or_sections": None,
            "dim": [1, 2, 3]
        }

    def input_conversion(self, input_shape, output_shape):
        node_constraint = dict()
        node_constraint["input_shape"] = input_shape
        return node_constraint
    
    '''The last two dimensions are split'''
    def get_possible_params(self, node_constraint=None, param_domain=None): 
        input_shape = node_constraint["input_shape"]
        results = list()
        for dim in range(2, 4):   #dim =[2,3]  i.e. dimension 3 and dimension 4
            max_divisor = min(input_shape[dim], 5) # FIXME: hard-coded the possible divisor. 
            for divisor in range(2, max_divisor):
                split_size_or_sections = input_shape[dim] // divisor #divisor<-max_divisor<=min(5,input_shape[dim 2or3])
                results.append({
                    "split_size_or_sections": split_size_or_sections,
                    "dim": dim
                    })
        return results
    
    '''
        in the corresponding dim, split into x*spilit_size_or_sections+remainder

        input_shape is [64, 3, 64, 64]
        result[0] is {'split_size_or_sections': 32, 'dim': 2}
        layout_info is [[64, 3, 32, 64], [64, 3, 32, 64]]
    
    
        input_shape is [64, 3, 64, 64]
        result[4] is {'split_size_or_sections': 21, 'dim': 3}
        layout_info is [[64, 3, 64, 21], [64, 3, 64, 21], [64, 3, 64, 21], [64, 3, 64, 1]]
    '''
    def get_layout_info(self, param, shape): 
        dim = param['dim']
        split_size_or_sections = param['split_size_or_sections']
        equal_divide_num = shape[dim] // split_size_or_sections
        last_size = shape[dim] % split_size_or_sections
        layout_info = list()
        single_shape = list()
        for t in range(4):
            if t != dim:
                single_shape.append(shape[t])
            else:
                single_shape.append(split_size_or_sections)
        layout_info = [single_shape for i in range(equal_divide_num)]
        if last_size != 0:
            remainder_shape = single_shape.copy()
            remainder_shape[dim] = last_size
            layout_info.append(remainder_shape)
        return layout_info
    
    def get_node(self, param):
        split_node = SplitWrapper(**param)  #instantiate SplitWrapper (a layer)
        cat_node = CatWrapper(dim=param["dim"]) #instantiate CatWrapper
        return (split_node, cat_node)

    def get_split_info(self):
        raise NotImplementedError

class ReshapeWrapper(nn.Module):
    def __init__(self, shape):
        super(ReshapeWrapper, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return torch.reshape(x, self.shape)
    
class WeightSum(nn.Module):
    def __init__(self, in_degree):
        super(WeightSum, self).__init__()
        self.in_degree = in_degree
        self.weights = nn.Parameter(torch.ones(self.in_degree), requires_grad=True)
    
    def forward(self, *x):
        out = 0
        for i in range(self.in_degree):
            out += x[i] * self.weights[i]
        return out

class OutputLayer(nn.Module):
    def __init__(self, input_shape, output_class, activation="ReLU"):
        super(OutputLayer, self).__init__()
        input_dim = 1
        for i in input_shape[1:]:
            input_dim *= i
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, output_class),
            nn.ReLU()
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # do the test
    conv2_node = Conv2d()
    conv2_node = ConvTranspose2d()
    
    # SeparableConv2d
    test_node = SeparableConv2d()
    node_constraint = {"H_in":224, "H_out":224, "C_in":36, "C_out":18, "W_in":224, "W_out":224}
    params = test_node.get_possible_params(node_constraint)
    nn_node = test_node.get_node(params[0])
    assert len(nn_node) == 2
    assert (nn_node(torch.ones(1, 36, 224, 224))).shape == (1, 36, 224, 224)
    
    # Bilinear
    test_node = Bilinear()
    node_constraint = test_node.input_conversion((1, 3, 224, 223), (1, 3, 224, 223))
    params = test_node.get_possible_params(node_constraint)
    nn_node = test_node.get_node(params[0])
    assert nn_node(torch.ones(1, 3, 224, 223)).shape == (1, 3, 224, 223)
    
    # BatchNorm2d
    test_node = BatchNorm2d()
    constraint = test_node.input_conversion((1, 3, 224, 224), (1, 3, 224, 224))
    params = test_node.get_possible_params(constraint)
    nn_node = test_node.get_node(params[0])
    assert nn_node(torch.ones(1, 3, 224, 223)).shape == (1, 3, 224, 223)
    
    # GroupNorm
    test_node = GroupNorm()
    constraint = test_node.input_conversion((1, 3, 224, 224), (1, 3, 224, 224))
    params = test_node.get_possible_params(constraint)
    nn_node = test_node.get_node(params[0])

    # Dropout
    test_node = Dropout()
    params = test_node.get_possible_params()
    nn_node = test_node.get_node(params[0])
    assert nn_node(torch.ones(1, 36, 224, 224)).shape == (1, 36, 224, 224)
    
    # SplitCat
    test_node = SplitCat()
    constraint = test_node.input_conversion((1, 3, 224, 224), (1, 3, 224, 224))
    params = test_node.get_possible_params(constraint)
    nn_node = test_node.get_node(params[0])
    layout_info = test_node.get_layout_info(params[0], (1, 3, 224, 224))
    assert len(nn_node) == 2
    assert len(layout_info) == 2
    
    
    
