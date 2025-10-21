
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def batchnorm3d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 100, 35, 45, 10).numpy()
    input_dict1 = {
        "num_features": 100,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(10, 50, 20, 25, 5).numpy()
    input_dict2 = {
        "num_features": 50,
        "eps": 0.01,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": False,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5, 25, 10, 12, 3).numpy()
    input_dict3 = {
        "num_features": 25,
        "eps": 1e-8,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": True,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(32, 64, 15, 18, 7).numpy()
    input_dict4 = {
        "num_features": 64,
        "eps": 1e-3,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(40, 32, 28, 32, 8).numpy()
    input_dict5 = {
        "num_features": 32,
        "eps": 1e-6,
        "momentum": 0.7,
        "affine": True,
        "track_running_stats": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(8, 128, 16, 20, 4).numpy()
    input_dict6 = {
        "num_features": 128,
        "eps": 1e-4,
        "momentum": 0.3,
        "affine": False,
        "track_running_stats": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(16, 64, 8, 10, 2).numpy()
    input_dict7 = {
        "num_features": 64,
        "eps": 1e-7,
        "momentum": 0.8,
        "affine": True,
        "track_running_stats": True,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict8 = {
        "num_features": 3,
        "eps": 1e-9,
        "momentum": 0.01,
        "affine": False,
        "track_running_stats": False,
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.nn.BatchNorm3d"] = batchnorm3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.BatchNorm3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BatchNorm3d'.")


check_valid('torch.nn.BatchNorm3d', generated_inputs['torch.nn.BatchNorm3d'], lib="torch", suffix=0)
