import itertools
from functools import reduce
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
import tvm.testing
import torch

def get_combine_list(param_dict):
    """ Sometimes there is no constraint for parameters.
    So we can just do the combinations on the parameter and
    return the result as possible params.
    
    Input: param_dict: dict of list.
        For example: {"inplace": [True, False]}
    Output: result: list of dict.
        For example: [{"inplace": True}, {"inplace": False}]
    """
    param_list = list()
    for key in param_dict:
        param_list.append(param_dict[key])
    product_results = list(itertools.product(*param_list))
    results = list()
    for result in product_results:
        r_dict = dict()
        for idx, key in enumerate(param_dict):
            r_dict[key] = result[idx]
        results.append(r_dict)
    return results

def get_factors(n):
    """Find the factos of given int.
    """
    result = list(([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))
    return set(reduce(list.__add__, result))

def get_random_input(shape_list):
    """Generate a list of random input by given shape_list
    """
    return tuple([torch.randn(i) for i in shape_list])


def onnx2relay(onnx_model, input_shapes=None, freeze_params=False):
    """This function turns the pytorch model into
    relay model. Notice that for now, every compiled
    model has fixed batch size (set in baseline_input)"""
    # Here we don't specify the input name.
    # As the input name must be consistent with ONNX original input name
    # which means we need to call onnxruntime to get input name.
    
    #if not isinstance(baseline_input, list):
    #    baseline_input = [baseline_input]
    #input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
    #input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
    if input_shapes == None:
        mod, params = relay.frontend.from_onnx(onnx_model, freeze_params=freeze_params)
    else:
        mod, params = relay.frontend.from_onnx(onnx_model, input_shapes, freeze_params=freeze_params)
    return mod, params

def build_relay(relay_graph, params, opt_level, target='llvm', dev=tvm.cpu(0), custom_opt_passes=None):
    """This function build relay graph and return 
    GraphRuntimeFactoryModule"""
    if custom_opt_passes is not None:
        with tvm.transform.PassContext(opt_level=4):
            relay_graph = custom_opt_passes(relay_graph)
        with tvm.transform.PassContext(opt_level=0):
            lib  = relay.build(relay_graph, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=opt_level):
            lib  = relay.build(relay_graph, target=target, params=params)
    return lib

def verify_compiled_model(lib, dev, baseline_input, baseline_outputs, rtol, atol, exec="normal", input_name=None, return_tensor=False):
    """This function takes an already compiled model as input.
    This can save the time if you want to test a fixed tvm runtime model
    with different input"""
    if not isinstance(baseline_input, list):
        baseline_input = [baseline_input]
    if exec == "normal":
        relay_model = graph_executor.GraphModule(lib["default"](dev))
    elif exec == "debug":
        relay_model = debug_executor.create(lib.get_graph_json(), lib.get_lib(), dev, dump_root="tvmdbg")
        relay_model.set_input(**lib.get_params())
    else:
        raise Exception("Graph executor not found")
    if input_name == None:
        compiled_input = dict(zip([i for i in range(len(baseline_input))], [inp.copy() for inp in baseline_input]))
    else:
        compiled_input = dict(zip(input_name, baseline_input))
    for name, inp in compiled_input.items():
        relay_model.set_input(name, inp)
    relay_model.run()
    tvm_output = list()
    pytorch_output = list()
    for i, baseline_output in enumerate(baseline_outputs):
        compiled_output = relay_model.get_output(i).asnumpy()
        if return_tensor == False:
            tvm.testing.assert_allclose(compiled_output, baseline_output, rtol=rtol, atol=atol)
        else:
            tvm_output.append(compiled_output)
            pytorch_output.append(baseline_output)
    del relay_model
    if return_tensor == True:
        return {"tvm": tvm_output, "pytorch": pytorch_output}

def torch2relay(torch_model, baseline_input, custom_convert_map={}):
    """This function turns the pytorch model into
    relay model"""
    if isinstance(baseline_input, torch.Tensor):
        baseline_input = [baseline_input]
        
    trace = torch.jit.trace(torch_model, [input.clone() for input in baseline_input])
    if isinstance(torch_model, torch.nn.Module):
        trace = trace.float().eval()
        
        # FIXME Disable GPU
        #if torch.cuda.is_available():
        #    trace = trace.cuda()
        #else:
        #    trace = trace.cpu()
    input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
    input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
    mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    return mod, params

