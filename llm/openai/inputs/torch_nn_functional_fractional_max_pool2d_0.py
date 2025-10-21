
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def fractional_max_pool2d_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(1, 3, 32, 32).astype(np.float32)
    kernel_size1 = (2, 2)
    output_size1 = (16, 16)
    output_ratio1 = (0.5, 0.5)
    return_indices1 = True
    
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "output_size": output_size1,
        "output_ratio": output_ratio1,
        "return_indices": return_indices1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(2, 3, 64, 64).astype(np.float32)
    kernel_size2 = (3, 3)
    output_size2 = (32, 32)
    output_ratio2 = (0.5, 0.5)
    return_indices2 = False
    
    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "output_size": output_size2,
        "output_ratio": output_ratio2,
        "return_indices": return_indices2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(1, 1, 64, 64).astype(np.float32)
    kernel_size3 = (2, 2)
    output_size3 = (32, 32)
    output_ratio3 = (0.5, 0.5)
    return_indices3 = True
    
    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "output_size": output_size3,
        "output_ratio": output_ratio3,
        "return_indices": return_indices3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(4, 2, 64, 64).astype(np.float32)
    kernel_size4 = (4, 4)
    output_size4 = (16, 16)
    output_ratio4 = (0.25, 0.25)
    return_indices4 = False

    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "output_size": output_size4,
        "output_ratio": output_ratio4,
        "return_indices": return_indices4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.nn.functional.fractional_max_pool2d"] = fractional_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.fractional_max_pool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.fractional_max_pool2d'.")


check_valid('torch.nn.functional.fractional_max_pool2d', generated_inputs['torch.nn.functional.fractional_max_pool2d'], lib="torch", suffix=0)
