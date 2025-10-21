
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def logical_xor_inputs():
    list_of_inputs = []

    input1 = torch.tensor([True, False, True, False]).numpy()
    other1 = torch.tensor([False, True, False, True]).numpy()
    out1 = torch.tensor([]).numpy()

    input_dict1 = {
        "input": input1,
        "other": other1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([[True, False], [False, True]]).numpy()
    other2 = torch.tensor([[False, True], [True, False]]).numpy()
    out2 = torch.tensor([]).numpy()

    input_dict2 = {
        "input": input2,
        "other": other2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.logical_xor"] = logical_xor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logical_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_xor'.")


check_valid('torch.logical_xor', generated_inputs['torch.logical_xor'], lib="torch", suffix=0)
