import torch
import torch.nn as nn
import onnx
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor

minimal_example = torch.nn.Sequential(
    torch.nn.LogSigmoid()
)
minimal_example.eval()
error_input = torch.load("logSoftmax_Error.pt")
error_input = torch.zeros([1, 12, 56, 28]).type(torch.float32)
error_input -= 90
with torch.no_grad():
    pytorch_result = minimal_example(error_input)

trace = torch.jit.trace(minimal_example, error_input)
TVM_input = [error_input.numpy()]
input_names = ["input{}".format(idx) for idx, inp in enumerate(TVM_input)]
input_shapes = list(zip(input_names, [inp.shape for inp in TVM_input]))
mod, params = tvm.relay.frontend.from_pytorch(trace, input_shapes)
target = "llvm"
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib  = relay.build(mod, target=target, params=params)
relay_model = graph_executor.GraphModule(lib["default"](dev))
compiled_input = dict(zip([i for i in range(len(TVM_input))], [inp.copy() for inp in TVM_input]))
for name, inp in compiled_input.items():
    relay_model.set_input(name, inp)
relay_model.run()
TVM_output = relay_model.get_output(0).asnumpy()

print("INF in pytorch: {}".format(np.any(np.isinf(pytorch_result.numpy()))))
print("INF in TVM: {}".format(np.any(np.isinf(TVM_output))))
