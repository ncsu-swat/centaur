
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bitwise_or_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3])
    other1 = torch.tensor([4, 5, 6])
    out1 = torch.empty_like(input1)
    
    input_dict1 = {
        "input": input1.numpy(),
        "other": other1.numpy(),
        "out": out1.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([[1, 0], [0, 1]])
    other2 = torch.tensor([[1, 1], [0, 0]])
    out2 = torch.empty_like(input2)
    
    input_dict2 = {
        "input": input2.numpy(),
        "other": other2.numpy(),
        "out": out2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([[-1, 2], [3, -4]])
    other3 = torch.tensor([[0, -1], [-2, 3]])
    out3 = torch.empty_like(input3)
    
    input_dict3 = {
        "input": input3.numpy(),
        "other": other3.numpy(),
        "out": out3.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([1, 1, 1, 1])
    other4 = torch.tensor([0, 0, 0, 0])
    out4 = torch.empty_like(input4)

    input_dict4 = {
        "input": input4.numpy(),
        "other": other4.numpy(),
        "out": out4.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.tensor([255, 0, 128])
    other5 = torch.tensor([0, 255, 128])
    out5 = torch.empty_like(input5)

    input_dict5 = {
        "input": input5.numpy(),
        "other": other5.numpy(),
        "out": out5.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([1])
    other6 = torch.tensor([0])
    out6 = torch.empty_like(input6)

    input_dict6 = {
        "input": input6.numpy(),
        "other": other6.numpy(),
        "out": out6.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    

    return list_of_inputs

generated_inputs["torch.bitwise_or"] = bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bitwise_or' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bitwise_or'.")


check_valid('torch.bitwise_or', generated_inputs['torch.bitwise_or'], lib="torch", suffix=0)
