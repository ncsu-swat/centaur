
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logical_or_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([True, False, True])
    other1 = torch.tensor([False, True, False])
    out1 = torch.empty_like(input1)
    
    input_dict1 = {
        "input": input1.numpy(),
        "other": other1.numpy(),
        "out": out1.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([[True, False], [False, True]])
    other2 = torch.tensor([[False, True], [True, False]])
    out2 = torch.empty_like(input2)
    
    input_dict2 = {
        "input": input2.numpy(),
        "other": other2.numpy(),
        "out": out2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([True, True, False, False])
    other3 = torch.tensor([False, False, True, True])
    out3 = torch.empty_like(input3)
    
    input_dict3 = {
        "input": input3.numpy(),
        "other": other3.numpy(),
        "out": out3.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.logical_or"] = logical_or_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logical_or' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_or'.")


check_valid('torch.logical_or', generated_inputs['torch.logical_or'], lib="torch", suffix=0)
