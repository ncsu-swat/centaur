
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def dropout2d_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(1, 3, 224, 224).numpy()
    input_dict1 = {
        "input": input1,
        "p": 0.5,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(2, 6, 128, 128).numpy()
    input_dict2 = {
        "input": input2,
        "p": 0.2,
        "training": False,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(4, 1, 64, 64).numpy()
    input_dict3 = {
        "input": input3,
        "p": 0.8,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(1, 16, 32, 32).numpy()
    input_dict4 = {
        "input": input4,
        "p": 0.0,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(3, 8, 16, 16).numpy()
    input_dict5 = {
        "input": input5,
        "p": 1.0,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(1, 32, 64, 64).numpy()
    input_dict6 = {
        "input": input6,
        "p": 0.5,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(2, 16, 32, 32).numpy()
    input_dict7 = {
        "input": input7,
        "p": 0.7,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(1, 64, 128, 128).numpy()
    input_dict8 = {
        "input": input8,
        "p": 0.3,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.randn(4, 3, 256, 256).numpy()
    input_dict9 = {
        "input": input9,
        "p": 0.6,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.randn(1, 128, 64, 64).numpy()
    input_dict10 = {
        "input": input10,
        "p": 0.9,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.dropout2d"] = dropout2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.dropout2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.dropout2d'.")


check_valid('torch.nn.functional.dropout2d', generated_inputs['torch.nn.functional.dropout2d'], lib="torch", suffix=0)
