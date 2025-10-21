
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def instancenorm3d_inputs():
    list_of_inputs = []

    input1 = np.random.rand(20, 100, 35, 45, 10).astype(np.float32)
    input_dict1 = {
        "num_features": 100,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float32,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(10, 50, 20, 25, 5).astype(np.float64)
    input_dict2 = {
        "num_features": 50,
        "eps": 1e-8,
        "momentum": 0.2,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float64,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(1, 256, 16, 16, 16).astype(np.float16)
    input_dict3 = {
        "num_features": 256,
        "eps": 1e-6,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float16,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(4, 128, 16, 16, 16).astype(np.float32)
    input_dict4 = {
        "num_features": 128,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.nn.InstanceNorm3d"] = instancenorm3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.InstanceNorm3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.InstanceNorm3d'.")


check_valid('torch.nn.InstanceNorm3d', generated_inputs['torch.nn.InstanceNorm3d'], lib="torch", suffix=0)
