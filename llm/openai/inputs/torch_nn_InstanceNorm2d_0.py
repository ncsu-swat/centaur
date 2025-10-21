
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def instancenorm2d_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(20, 100, 35, 45).astype(np.float32)
    input_dict1 = {
        "num_features": 100,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(10, 50, 20, 20).astype(np.float32)
    input_dict2 = {
        "num_features": 50,
        "eps": 1e-8,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(1, 64, 10, 10).astype(np.float32)
    input_dict3 = {
        "num_features": 64,
        "eps": 1e-6,
        "momentum": 0.5,
        "affine": False,
        "track_running_stats": True,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(32, 3, 64, 64).astype(np.float32)
    input_dict4 = {
        "num_features": 3,
        "eps": 1e-7,
        "momentum": 0.2,
        "affine": True,
        "track_running_stats": False,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.rand(8, 128, 128, 128).astype(np.float32)
    input_dict5 = {
        "num_features": 128,
        "eps": 1e-4,
        "momentum": 0.3,
        "affine": False,
        "track_running_stats": True,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(16, 64, 32, 32).astype(np.float32)
    input_dict6 = {
        "num_features": 64,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": False,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(4, 256, 64, 64).astype(np.float32)
    input_dict7 = {
        "num_features": 256,
        "eps": 1e-3,
        "momentum": 0.7,
        "affine": False,
        "track_running_stats": True,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(32, 16, 64, 64).astype(np.float32)
    input_dict8 = {
        "num_features": 16,
        "eps": 1e-9,
        "momentum": 0.4,
        "affine": True,
        "track_running_stats": False,
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.random.rand(2, 32, 16, 16).astype(np.float32)
    input_dict9 = {
        "num_features": 32,
        "eps": 1e-2,
        "momentum": 0.6,
        "affine": False,
        "track_running_stats": True,
        "input": input9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.random.rand(64, 1, 32, 32).astype(np.float32)
    input_dict10 = {
        "num_features": 1,
        "eps": 1e-1,
        "momentum": 0.8,
        "affine": True,
        "track_running_stats": False,
        "input": input10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.InstanceNorm2d"] = instancenorm2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.InstanceNorm2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.InstanceNorm2d'.")


check_valid('torch.nn.InstanceNorm2d', generated_inputs['torch.nn.InstanceNorm2d'], lib="torch", suffix=0)
