
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def group_norm_inputs():
    list_of_inputs = []

    input1 = np.random.rand(20, 6, 10, 10).astype(np.float32)
    input_dict1 = {
        "num_groups": 3,
        "num_channels": 6,
        "eps": 1e-5,
        "affine": True,
        "dtype": np.float32,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(10, 32, 8, 8).astype(np.float64)
    input_dict2 = {
        "num_groups": 8,
        "num_channels": 32,
        "eps": 1e-8,
        "affine": False,
        "dtype": np.float64,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.nn.GroupNorm"] = group_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.GroupNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GroupNorm'.")


check_valid('torch.nn.GroupNorm', generated_inputs['torch.nn.GroupNorm'], lib="torch", suffix=0)
