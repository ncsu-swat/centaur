
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hardtanh_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict1 = {
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False,
        "min_value": None,
        "max_value": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(5).astype(np.float32)
    input_dict2 = {
        "min_val": -0.5,
        "max_val": 0.5,
        "inplace": True,
        "min_value": None,
        "max_value": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(1).astype(np.float32)
    input_dict3 = {
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False,
        "min_value": None,
        "max_value": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(2, 2).astype(np.float32)
    input_dict4 = {
        "min_val": -2.0,
        "max_val": 2.0,
        "inplace": True,
        "min_value": None,
        "max_value": None,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(3, 3, 3).astype(np.float32)
    input_dict5 = {
        "min_val": 0.0,
        "max_val": 1.0,
        "inplace": False,
        "min_value": None,
        "max_value": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.nn.Hardtanh"] = hardtanh_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Hardtanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Hardtanh'.")


check_valid('torch.nn.Hardtanh', generated_inputs['torch.nn.Hardtanh'], lib="torch", suffix=0)
