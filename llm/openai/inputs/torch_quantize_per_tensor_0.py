
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def quantize_per_tensor_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scale1 = 0.1
    zero_point1 = 0
    dtype1 = torch.int8
    
    input_dict1 = {
        "input": input1,
        "scale": scale1,
        "zero_point": zero_point1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale2 = 0.5
    zero_point2 = 128
    dtype2 = torch.int8
    
    input_dict2 = {
        "input": input2,
        "scale": scale2,
        "zero_point": zero_point2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.0, -1.0, 2.0, -3.0], dtype=np.float32)
    scale3 = 1.0
    zero_point3 = -10
    dtype3 = torch.int8

    input_dict3 = {
        "input": input3,
        "scale": scale3,
        "zero_point": zero_point3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.quantize_per_tensor"] = quantize_per_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.quantize_per_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.quantize_per_tensor'.")


check_valid('torch.quantize_per_tensor', generated_inputs['torch.quantize_per_tensor'], lib="torch", suffix=0)
