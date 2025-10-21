
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cummin_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dim1 = 0
    
    
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dim2 = 1
    
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.cummin"] = cummin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cummin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cummin'.")


check_valid('torch.cummin', generated_inputs['torch.cummin'], lib="torch", suffix=0)
