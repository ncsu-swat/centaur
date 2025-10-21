
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_full_inputs():
    list_of_inputs = []
    
    input1 = (2, 3)
    fill_value1 = 3.141592
    out1 = np.zeros((2, 3), dtype=np.float64)
    dtype1 = torch.float64
    requires_grad1 = False
    
    input_dict1 = {
        "size": input1,
        "fill_value": fill_value1,
        "out": out1,
        "dtype": dtype1,
        "requires_grad": requires_grad1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = (5,)
    fill_value2 = -1.0
    out2 = np.zeros(5, dtype=np.float32)
    dtype2 = torch.float32
    requires_grad2 = True
    
    input_dict2 = {
        "size": input2,
        "fill_value": fill_value2,
        "out": out2,
        "dtype": dtype2,
        "requires_grad": requires_grad2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.full"] = torch_full_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.full' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.full'.")


check_valid('torch.full', generated_inputs['torch.full'], lib="torch", suffix=0)
