
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bitwise_left_shift_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3]).numpy()
    other1 = torch.tensor([1, 2, 3]).numpy()
    out1 = torch.tensor(np.zeros_like(input1)).numpy()
    
    input_dict1 = {
        "input": input1,
        "other": other1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([[-1, -2], [3, 4]]).numpy()
    other2 = torch.tensor([2, 3]).numpy()
    out2 = torch.tensor(np.zeros_like(input2)).numpy()
    
    input_dict2 = {
        "input": input2,
        "other": other2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.tensor([0, 1, 2, 3]).numpy()
    other3 = torch.tensor([4]).numpy()
    out3 = torch.tensor(np.zeros_like(input3)).numpy()
    
    input_dict3 = {
        "input": input3,
        "other": other3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.tensor([5, 10, 15]).numpy()
    other4 = torch.tensor([0, 1, 2]).numpy()
    out4 = torch.tensor(np.zeros_like(input4)).numpy()
    
    input_dict4 = {
        "input": input4,
        "other": other4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.tensor([[1, 2], [3, 4]]).numpy()
    other5 = torch.tensor([[1, 1], [2, 2]]).numpy()
    out5 = torch.tensor(np.zeros_like(input5)).numpy()
    
    input_dict5 = {
        "input": input5,
        "other": other5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.bitwise_left_shift"] = bitwise_left_shift_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bitwise_left_shift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_left_shift'.")


check_valid('torch.bitwise_left_shift', generated_inputs['torch.bitwise_left_shift'], lib="torch", suffix=0)
