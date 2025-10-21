
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def max_pool2d_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(1, 3, 32, 32).astype(np.float32)
    kernel_size1 = 2
    stride1 = (1, 1)
    padding1 = (0, 0)
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
    
    input2 = np.random.rand(2, 3, 16, 16).astype(np.float32)
    kernel_size2 = 3
    stride2 = (2, 2)
    padding2 = (1, 1)
    dilation2 = 2
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

    input3 = np.random.rand(1, 1, 64, 64).astype(np.float32)
    kernel_size3 = 5
    stride3 = (3, 3)
    padding3 = (2, 2)
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

    input4 = np.random.rand(4, 2, 8, 8).astype(np.float32)
    kernel_size4 = 1
    stride4 = (1, 1)
    padding4 = (0, 0)
    dilation4 = 3
    return_indices4 = False
    ceil_mode4 = False
    
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

    input5 = np.random.rand(1, 4, 128, 64).astype(np.float32)
    kernel_size5 = 2
    stride5 = (2, 1)
    padding5 = (1, 0)
    dilation5 = 1
    return_indices5 = True
    ceil_mode5 = True

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

    input6 = np.random.rand(2, 1, 32, 32).astype(np.float32)
    kernel_size6 = 4
    stride6 = (4, 4)
    padding6 = (0, 0)
    dilation6 = 1
    return_indices6 = False
    ceil_mode6 = False

    input_dict6 = {
        "input": input6,
        "kernel_size": kernel_size6,
        "stride": stride6,
        "padding": padding6,
        "dilation": dilation6,
        "return_indices": return_indices6,
        "ceil_mode": ceil_mode6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(1, 3, 64, 64).astype(np.float32)
    kernel_size7 = 3
    stride7 = (1, 1)
    padding7 = (1, 1)
    dilation7 = 2
    return_indices7 = True
    ceil_mode7 = False

    input_dict7 = {
        "input": input7,
        "kernel_size": kernel_size7,
        "stride": stride7,
        "padding": padding7,
        "dilation": dilation7,
        "return_indices": return_indices7,
        "ceil_mode": ceil_mode7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(3, 1, 16, 16).astype(np.float32)
    kernel_size8 = 5
    stride8 = (2, 2)
    padding8 = (2, 2)
    dilation8 = 1
    return_indices8 = False
    ceil_mode8 = True

    input_dict8 = {
        "input": input8,
        "kernel_size": kernel_size8,
        "stride": stride8,
        "padding": padding8,
        "dilation": dilation8,
        "return_indices": return_indices8,
        "ceil_mode": ceil_mode8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(1, 5, 32, 32).astype(np.float32)
    kernel_size9 = 2
    stride9 = (1, 1)
    padding9 = (0, 0)
    dilation9 = 3
    return_indices9 = True
    ceil_mode9 = False

    input_dict9 = {
        "input": input9,
        "kernel_size": kernel_size9,
        "stride": stride9,
        "padding": padding9,
        "dilation": dilation9,
        "return_indices": return_indices9,
        "ceil_mode": ceil_mode9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 4, 64, 64).astype(np.float32)
    kernel_size10 = 3
    stride10 = (3, 3)
    padding10 = (1, 1)
    dilation10 = 1
    return_indices10 = False
    ceil_mode10 = True

    input_dict10 = {
        "input": input10,
        "kernel_size": kernel_size10,
        "stride": stride10,
        "padding": padding10,
        "dilation": dilation10,
        "return_indices": return_indices10,
        "ceil_mode": ceil_mode10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool2d_3"] = max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool2d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool2d_3'.")


check_valid('torch.nn.functional.max_pool2d', generated_inputs['torch.nn.functional.max_pool2d_3'], lib="torch", suffix=3)
