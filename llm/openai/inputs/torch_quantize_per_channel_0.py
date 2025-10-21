
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def quantize_per_channel_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    scales1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_points1 = np.array([1, 2, 3], dtype=np.int8)
    axis1 = 1
    dtype1 = torch.int8

    input_dict1 = {
        "input": input1,
        "scales": scales1,
        "zero_points": zero_points1,
        "axis": axis1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    scales1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    zero_points1 = np.array([1, 2, 3], dtype=np.int8)
    axis1 = 0
    dtype1 = torch.int8

    input_dict1 = {
        "input": input1,
        "scales": scales1,
        "zero_points": zero_points1,
        "axis": axis1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.quantize_per_channel"] = quantize_per_channel_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.quantize_per_channel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.quantize_per_channel'.")


check_valid('torch.quantize_per_channel', generated_inputs['torch.quantize_per_channel'], lib="torch", suffix=0)
