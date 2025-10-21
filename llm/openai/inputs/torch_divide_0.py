
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def divide_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other1 = torch.tensor([2.0, 4.0, 6.0]).numpy()
    rounding_mode1 = "trunc"
    out1 = torch.tensor([])
    input_dict1 = {
        "input": input1,
        "other": other1,
        "rounding_mode": rounding_mode1,
        "out": out1.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 2).numpy()
    other2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    rounding_mode2 = "floor"
    out2 = torch.tensor([])
    input_dict2 = {
        "input": input2,
        "other": other2,
        "rounding_mode": rounding_mode2,
        "out": out2.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([-1.0, 0.0, 1.0]).numpy()
    other3 = torch.tensor([1.0, 1.0, 1.0]).numpy()
    rounding_mode3 = "floor"
    out3 = torch.tensor([])
    input_dict3 = {
        "input": input3,
        "other": other3,
        "rounding_mode": rounding_mode3,
        "out": out3.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([2.0, 4.0, 6.0]).numpy()
    other4 = torch.tensor([0.5, 1.0, 1.5]).numpy()
    rounding_mode4 = None
    out4 = torch.tensor([])
    input_dict4 = {
        "input": input4,
        "other": other4,
        "rounding_mode": rounding_mode4,
        "out": out4.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(3, 4, 5).numpy()
    other5 = torch.tensor(np.random.rand(3, 4, 5)).numpy()
    rounding_mode5 = "trunc"
    out5 = torch.tensor([])
    input_dict5 = {
        "input": input5,
        "other": other5,
        "rounding_mode": rounding_mode5,
        "out": out5.numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.divide"] = divide_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.divide'.")


check_valid('torch.divide', generated_inputs['torch.divide'], lib="torch", suffix=0)
