import tvm
from tvm import relay
import re
import functools
from inspect import getmembers, isfunction

'''
seq = tvm.transform.Sequential(
        [
            transform.SimplifyInference(),
            #transform.FuseOps()
        ]
    )
seq = tvm.transform.Sequential(global_variable.default_pass_order[:20])
lib = tvm_build_model(mod, param, seq_opt_level=3, build_opt_level=3, seq=seq)
'''

def tvm_build_model(mod, param, seq_opt_level=4, build_opt_level=0, seq=None):
    """Build the model by given sequence.
    
    Parameters
    ----------
    mod : tvm.ir.module.IRModule
    
    dict : dict
    
    seq_opt_level : int
    
    build_opt_level : int
    
    seq : tvm.ir.transform.Sequential
    
    Returns
    -------
    lib : GraphRuntimeFactoryModule
    """
    print("begin building the model")
    if seq != None:
        with tvm.transform.PassContext(opt_level=seq_opt_level):
            mod = seq(mod)
            print("transformation done")
    with tvm.transform.PassContext(opt_level=build_opt_level):
        print("building the model under opt_level:", tvm.transform.PassContext.current().opt_level)
        lib  = relay.build(mod, target='llvm', params=param)
    return lib

def get_all_passes_tvm():
    """Iterate through all the members of tvm.relay.transform,
    check those who have pass_info (i.e. who is pass),
    and return the opt_level and required_pass info of the passes.

    Returns
    -------
    pass_by_name : dict
        the dict that contains the pass info.
    
    """
    pass_by_name = dict()
    # only check those who are functions
    for member in getmembers(tvm.relay.transform, isfunction):
        try:
            candidate_function = member[1]
            opt_pass = candidate_function()
        except Exception as e:
            continue
        if hasattr(opt_pass, "pass_info"):
            if (opt_pass.pass_info.name == 'sequential'):
                # some pass are made of other passes
                # in the form of sequential
                pass_name = candidate_function.__name__
            else:
                pass_name = opt_pass.pass_info.name
            pass_name = pass_name.strip()
            opt_level = opt_pass.pass_info.opt_level
            required_pass = opt_pass.pass_info.required
            pass_by_name[pass_name] = {'opt_level': opt_level, 
                                      'required_pass': required_pass}
    return pass_by_name

def encode_passses_into_grams(pass_by_name):
    result_encoding = dict()
    for name in pass_by_name:
        result_encoding[name] = name.lower()
    return result_encoding


# We now use following code to do the profiling
'''
from tvm.ir.instrument import (
    PassTimingInstrument,
    pass_instrument,
)
timing_inst = PassTimingInstrument()
with tvm.transform.PassContext(opt_level=4, instruments=[timing_inst]):
    mod = tvm.transform.Sequential(passes=[x() for x in _RELAY_FUNCTION_HARD_PASSES_], opt_level=4)(mod)
    profiles = timing_inst.render()
'''

"""
Legacy usage, see https://github.com/apache/tvm/pull/7500/files
"""


'''
global_profiles = ''
def pass_profiler_tvm(func):
    """This is a decorator for profiling TVM passes.
    Use this to decorate any functions which have graph building.
    Retrieve the profiling result using global variable: global_profiles
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tvm.transform.enable_pass_profiling()
        tvm.transform.clear_pass_profiles()
        result = func(*args, **kwargs)
        global global_profiles
        global_profiles = tvm.transform.render_pass_profiles()
        tvm.transform.clear_pass_profiles()
        tvm.transform.disable_pass_profiling()
        return result
    return wrapper
'''

def get_profile_indent1_tvm(profiles):
    """Parse the profiling result of TVM render_pass_profiles
    Notice when an optimization consists multiple sub-optimization,
    render_pass_profiles will use '\t' to indicate the nested structure.
    This function returns the optmization sequence with depth at most = 2.
    
    Input format example: InferType: 242us [242us] (0.03%; 0.03%)
        passName:exec_time:exec_time exclude sub pass:(percent of total exec time, 
            percent of parent pass exec time) 

    Parameters
    ----------
    profiles : tvm.runtime.container.String
        the profiling string returned by 
        tvm.transform.render_pass_profiles
    
    Returns
    -------
    result_list : dict. 
        the result list
    """
    result_list = list()
    sequential_id = 0
    indent1 = False
    for line in profiles.split('\n')[:-1]:
        name = line.split(':')[0]
        exec_time_str_list = re.findall("\s[0-9].*us\s", line)
        assert len(exec_time_str_list) == 1
        exec_time = exec_time_str_list[0].strip()
        exec_proportion_str_list = re.findall("\(.*;.*\)", line)
        assert len(exec_proportion_str_list) == 1
        exec_proportion = re.findall("\(.*;.*\)", line)[0].strip()
        if line.count('\t') == 0: # outer sequence
            if indent1 == True:
                result_list.append(temp_dict)
                indent1 = False
            if name == 'sequential':
                sequence_name = 'sequential{}'.format(sequential_id)
                sequential_id += 1
                temp_dict = {
                    sequence_name: {
                        "sub":[], 
                        "exec":(exec_time, exec_proportion)
                    }
                }# use dict to record the sub-sequence
                indent1 = True
            else:
                result_list.append((name, exec_time, exec_proportion))
        if line.count('\t') == 1 and indent1: # indent = 1 sequence
            temp_dict[sequence_name]["sub"].append((name[1:], exec_time, exec_proportion)) 
    return result_list

