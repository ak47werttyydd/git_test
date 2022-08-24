"""This file contains the pytorch model importation function
Reference: tests/python/frontend/pytorch/test_forward.py"""

import tvm
import torchvision
import torch
import os
import sys
from tvm import relay
import tvm.testing
from tvm.contrib import graph_runtime, graph_executor

sys.path.insert(0,'..')
from pass_analysis_util import pass_profiler_tvm

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

torch.backends.cudnn.deterministic = True

def load_torchvision(model_name):
    """Given a model name, returns a Torchvision model in eval mode as well
    as an example input."""
    with torch.no_grad():
        if model_name.startswith("inception"):
            height = width = 299
            mean = [0.5, 0.5, 0.5] # Imagenet
            std = [0.5, 0.5, 0.5] # Imagenet
        else:
            height = width = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        input_shape = [1, 3, height, width]
        input_data = torch.randn(input_shape).float()
        for channel in range(3):
            input_data[:, channel] -= mean[channel]
            input_data[:, channel] /= std[channel]

        if model_name.startswith("googlenet"):
            model = getattr(torchvision.models, model_name)(pretrained=True, aux_logits=True)
        else:
            model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.float().eval()
        return model, [input_data]

def load_model(model_name):
    """Given a model name, returns a model as well as an example input."""
    if hasattr(torchvision.models, model_name):
        return load_torchvision(model_name)
    try:
        import pretrainedmodels

        if hasattr(pretrainedmodels, model_name):
            return load_pretrainedmodels(model_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install pretrainedmodels.pytorch")
    raise RuntimeError("Model not supported")

def assert_shapes_match(tru, est):
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))
        

def torch2relay(torch_model, baseline_input, custom_convert_map={}):
    """This function turns the pytorch model into
    relay model"""
    if isinstance(baseline_input, torch.Tensor):
        baseline_input = [baseline_input]
        
    trace = torch.jit.trace(torch_model, [input.clone() for input in baseline_input])
    if isinstance(torch_model, torch.nn.Module):
        trace = trace.float().eval()

        if torch.cuda.is_available():
            trace = trace.cuda()
        else:
            trace = trace.cpu()
    input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
    input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
    mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    return mod, params

def get_baseline_outputs(baseline_model, baseline_input):
    """Get baseline output by forwarding the baseline_input 
    in the baseline_model"""

    if isinstance(baseline_input, torch.Tensor):
        baseline_input = [baseline_input]

    if torch.cuda.is_available():
        if isinstance(baseline_model, torch.nn.Module):
            baseline_model = baseline_model.cuda()
        baseline_input = [inp.cuda() for inp in baseline_input]  

    with torch.no_grad():
        baseline_outputs = baseline_model(*[input.clone() for input in baseline_input])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)
    return baseline_outputs

def verify_compiled_model(lib, dev, baseline_input, baseline_outputs, rtol, atol):
    """This function takes an already compiled model as input.
    This can save the time if you want to test a fixed tvm runtime model
    with different input"""
    if isinstance(baseline_input, torch.Tensor):
        baseline_input = [baseline_input]
    relay_model = graph_executor.GraphModule(lib["default"](dev))
    input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
    compiled_input = dict(zip(input_names, [inp.clone().cpu().numpy() for inp in baseline_input]))
    for name, inp in compiled_input.items():
        relay_model.set_input(name, inp)
    relay_model.run()
    for i, baseline_output in enumerate(baseline_outputs):
        compiled_output = relay_model.get_output(i).asnumpy()
        assert_shapes_match(baseline_output, compiled_output)
        tvm.testing.assert_allclose(compiled_output, baseline_output, rtol=rtol, atol=atol)

def verify_model(model_name, input_data=[], mod=None, params=None, opt_level=3, custom_convert_map={}, rtol=1e-5, atol=1e-5):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    if isinstance(model_name, str):
        baseline_model, baseline_input = load_model(model_name)
    elif isinstance(input_data, list):
        baseline_model = model_name
        baseline_input = input_data
    elif isinstance(input_data, torch.Tensor) or len(input_data.shape) == 0:
        baseline_model = model_name
        baseline_input = [input_data]
    else:
        assert False, "Unexpected input format"
    
    if not (mod != None and params != None):
        mod, params = torch2relay(baseline_model, baseline_input, custom_convert_map)
    input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
    for arg in mod["main"].params[: len(input_names)]:
        assert arg.name_hint in input_names
    #compiled_input = dict(zip(input_names, [inp.clone().cpu().numpy() for inp in baseline_input]))
    baseline_outputs = get_baseline_outputs(baseline_model, baseline_input) 
    with tvm.transform.PassContext(opt_level=opt_level):
        #for target, dev in tvm.testing.enabled_targets():
        # FIXME: we now only focus on llvm target.
        target = 'llvm'
        dev = tvm.cpu(0)
        lib  = relay.build(mod, target=target, params=params)
        verify_compiled_model(lib, dev, baseline_input, baseline_outputs, rtol, atol)

    del model_name
    del baseline_model
    torch.cuda.empty_cache()

def get_parameter_num_torch(model):
    """calculate the total numer of parameters in a given pytorch model.
    """
    return sum(p.numel() for p in model.parameters())

#@pass_profiler_tvm
def test_resnet18():
    torch.set_grad_enabled(False)
    verify_model("resnet18", atol=1e-4, rtol=1e-4)


#@pass_profiler_tvm
def test_squeezenet1_0():
    torch.set_grad_enabled(False)
    verify_model("squeezenet1_0", atol=1e-4, rtol=1e-4)


#@pass_profiler_tvm
def test_squeezenet1_1():
    torch.set_grad_enabled(False)
    verify_model("squeezenet1_1", atol=1e-4, rtol=1e-4)


#@pass_profiler_tvm
def test_densenet121():
    torch.set_grad_enabled(False)
    verify_model("densenet121", atol=1e-4, rtol=1e-4)


#@pass_profiler_tvm
def test_inception_v3():
    torch.set_grad_enabled(False)
    verify_model("inception_v3", atol=1e-4, rtol=1e-4)


#@pass_profiler_tvm
def test_googlenet():
    torch.set_grad_enabled(False)
    verify_model("googlenet", atol=1e-4, rtol=1e-4)


#@pass_profiler_tvm
def test_mnasnet0_5():
    torch.set_grad_enabled(False)
    verify_model("mnasnet0_5", atol=1e-4, rtol=1e-4)


#@pass_profiler_tvm
def test_mobilenet_v2():
    torch.set_grad_enabled(False)
    verify_model("mobilenet_v2", atol=1e-4, rtol=1e-4)