
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def max_pool2d_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(1, 3, 32, 32).astype(np.float32)
    kernel_size1 = 2
    stride1 = 1
    padding1 = 0
    dilation1 = 1
    return_indices1 = True
    ceil_mode1 = False
    
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

    input2 = np.random.rand(4, 1, 16, 16).astype(np.float32)
    kernel_size2 = 3
    stride2 = 2
    padding2 = 1
    dilation2 = 1
    return_indices2 = False
    ceil_mode2 = True
    
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
    
    input3 = np.random.rand(2, 3, 8, 8).astype(np.float32)
    kernel_size3 = 2
    stride3 = 2
    padding3 = 0
    dilation3 = 2
    return_indices3 = True
    ceil_mode3 = False
    
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

    input4 = np.random.rand(1, 2, 64, 64).astype(np.float32)
    kernel_size4 = 4
    stride4 = 4
    padding4 = 1
    dilation4 = 1
    return_indices4 = False
    ceil_mode4 = True
    
    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "dilation": dilation4,
        "return_indices": return_indices4,
        "ceil_mode": ceil_mode4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.rand(8, 1, 32, 32).astype(np.float32)
    kernel_size5 = 1
    stride5 = 1
    padding5 = 0
    dilation5 = 1
    return_indices5 = True
    ceil_mode5 = False
    
    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "dilation": dilation5,
        "return_indices": return_indices5,
        "ceil_mode": ceil_mode5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    

    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool2d_1"] = max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool2d_1'.")


check_valid('torch.nn.functional.max_pool2d', generated_inputs['torch.nn.functional.max_pool2d_1'], lib="torch", suffix=1)
