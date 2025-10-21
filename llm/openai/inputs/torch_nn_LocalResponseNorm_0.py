
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def local_response_norm_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(32, 5, 24, 24).numpy()
    input_dict1 = {
        "size": 2,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(16, 5, 7, 7, 7, 7).numpy()
    input_dict2 = {
        "size": 3,
        "alpha": 0.0002,
        "beta": 0.8,
        "k": 1.5,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(1, 3, 32, 32).numpy()
    input_dict3 = {
        "size": 1,
        "alpha": 0.00005,
        "beta": 0.7,
        "k": 0.5,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(64, 10, 16, 16).numpy()
    input_dict4 = {
        "size": 5,
        "alpha": 0.00015,
        "beta": 0.78,
        "k": 1.2,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(8, 8, 64, 64).numpy()
    input_dict5 = {
        "size": 4,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 2, 2, 2).numpy()
    input_dict6 = {
        "size": 1,
        "alpha": 0.001,
        "beta": 1.0,
        "k": 2.0,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(16, 3, 128, 128).numpy()
    input_dict7 = {
        "size": 2,
        "alpha": 0.00001,
        "beta": 0.65,
        "k": 0.8,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(32, 1, 64, 64).numpy()
    input_dict8 = {
        "size": 3,
        "alpha": 0.0002,
        "beta": 0.9,
        "k": 1.1,
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.randn(4, 4, 32, 32).numpy()
    input_dict9 = {
        "size": 1,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
        "input": input9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.randn(1, 1, 1, 1).numpy()
    input_dict10 = {
        "size": 1,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
        "input": input10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.LocalResponseNorm"] = local_response_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LocalResponseNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LocalResponseNorm'.")


check_valid('torch.nn.LocalResponseNorm', generated_inputs['torch.nn.LocalResponseNorm'], lib="torch", suffix=0)
