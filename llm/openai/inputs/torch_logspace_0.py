
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logspace_inputs():
    list_of_inputs = []
    
    input1 = {'start': -10.0, 'end': 10.0, 'steps': 5, 'base': 10.0, 'out': np.array([]), 'dtype': torch.float64, 'requires_grad': False}
    list_of_inputs.append(copy.deepcopy(input1))
    
    input2 = {'start': 0.1, 'end': 1.0, 'steps': 5, 'base': 10.0, 'out': np.array([]), 'dtype': torch.float64, 'requires_grad': True}
    list_of_inputs.append(copy.deepcopy(input2))
    
    input3 = {'start': 0.1, 'end': 1.0, 'steps': 1, 'base': 10.0, 'out': np.array([]), 'dtype': torch.float64, 'requires_grad': False}
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {'start': 2.0, 'end': 2.0, 'steps': 1, 'base': 2.0, 'out': np.array([]), 'dtype': torch.float64, 'requires_grad': False}
    list_of_inputs.append(copy.deepcopy(input4))
    
    input5 = {'start': -5.0, 'end': 5.0, 'steps': 10, 'base': 2.0, 'out': np.array([]), 'dtype': torch.float64, 'requires_grad': True}
    list_of_inputs.append(copy.deepcopy(input5))
    
    input6 = {'start': 1.0, 'end': 100.0, 'steps': 100, 'base': np.e, 'out': np.array([]), 'dtype': torch.float64, 'requires_grad': False}
    list_of_inputs.append(copy.deepcopy(input6))
    
    input7 = {'start': -1.0, 'end': 1.0, 'steps': 3, 'base': 0.5, 'out': np.array([]), 'dtype': torch.float64, 'requires_grad': True}
    list_of_inputs.append(copy.deepcopy(input7))

    input8 = {'start': 5.0, 'end': 5.0, 'steps': 5, 'base': 1.0, 'out': np.array([]), 'dtype': torch.float64, 'requires_grad': False}
    list_of_inputs.append(copy.deepcopy(input8))
    
    input9 = {'start': -2.0, 'end': 0.0, 'steps': 4, 'base': 10.0, 'out': np.array([]), 'dtype': torch.float64, 'requires_grad': False}
    list_of_inputs.append(copy.deepcopy(input9))
    

    return list_of_inputs

generated_inputs["torch.logspace"] = logspace_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logspace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logspace'.")


check_valid('torch.logspace', generated_inputs['torch.logspace'], lib="torch", suffix=0)
