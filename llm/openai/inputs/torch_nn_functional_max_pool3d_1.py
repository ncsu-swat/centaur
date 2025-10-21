
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def max_pool3d_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(1, 1, 3, 3, 3).astype(np.float32)
    kernel_size1 = 2
    stride1 = 1
    padding1 = 0
    dilation1 = 1
    return_indices1 = True
    ceil_mode1 = True
    
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "return_indices": return_indices1,
        "ceil_mode": ceil_mode1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(1, 3, 3, 3, 3).astype(np.float32)
    kernel_size2 = 2
    stride2 = 2
    padding2 = 0
    dilation2 = 1
    return_indices2 = False
    ceil_mode2 = False
    
    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "dilation": dilation2,
        "return_indices": return_indices2,
        "ceil_mode": ceil_mode2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(2, 2, 2, 2, 2).astype(np.float32)
    kernel_size3 = 2
    stride3 = 1
    padding3 = 0
    dilation3 = 1
    return_indices3 = True
    ceil_mode3 = True
    
    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "dilation": dilation3,
        "return_indices": return_indices3,
        "ceil_mode": ceil_mode3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    
    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool3d_1"] = max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool3d_1'.")


check_valid('torch.nn.functional.max_pool3d', generated_inputs['torch.nn.functional.max_pool3d_1'], lib="torch", suffix=1)
