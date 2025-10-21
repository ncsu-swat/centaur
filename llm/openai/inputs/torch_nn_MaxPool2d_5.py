
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def maxpool2d_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(20, 16, 50, 32).numpy()
    input_dict1 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(10, 3, 64, 64).numpy()
    input_dict2 = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": 1,
        "dilation": (1, 2),
        "return_indices": True,
        "ceil_mode": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(1, 1, 100, 100).numpy()
    input_dict3 = {
        "kernel_size": 5,
        "stride": 5,
        "padding": 2,
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(4, 8, 32, 32).numpy()
    input_dict4 = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(8, 4, 256, 256).numpy()
    input_dict5 = {
        "kernel_size": 3,
        "stride": 3,
        "padding": 1,
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 1, 64, 64).numpy()
    input_dict6 = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_5"] = maxpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool2d_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool2d_5'.")


check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_5'], lib="torch", suffix=5)
