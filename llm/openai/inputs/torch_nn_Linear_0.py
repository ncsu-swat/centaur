
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def linear_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(128, 20).astype(np.float32)
    input_dict1 = {
        'in_features': 20,
        'out_features': 30,
        'bias': True,
        'dtype': torch.float32,
        'input': input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(64, 10).astype(np.float64)
    input_dict2 = {
        'in_features': 10,
        'out_features': 5,
        'bias': False,
        'dtype': torch.float64,
        'input': input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(32, 5).astype(np.float16)
    input_dict3 = {
        'in_features': 5,
        'out_features': 2,
        'bias': True,
        'dtype': torch.float16,
        'input': input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1, 1).astype(np.float32)
    input_dict4 = {
        'in_features': 1,
        'out_features': 1,
        'bias': True,
        'dtype': torch.float32,
        'input': input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    return list_of_inputs

generated_inputs["torch.nn.Linear"] = linear_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Linear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Linear'.")


check_valid('torch.nn.Linear', generated_inputs['torch.nn.Linear'], lib="torch", suffix=0)
