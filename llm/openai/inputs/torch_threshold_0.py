
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def threshold_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1.0, -2.0, 3.0, -4.0]).numpy()
    threshold1 = 0.0
    value1 = 5.0
    input_dict1 = {
        "input": input1,
        "threshold": threshold1,
        "value": value1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(2, 2).numpy()
    threshold2 = 1.0
    value2 = -1.0
    input_dict2 = {
        "input": input2,
        "threshold": threshold2,
        "value": value2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    threshold3 = -0.5
    value3 = 10.0
    input_dict3 = {
        "input": input3,
        "threshold": threshold3,
        "value": value3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([0.5, 0.2, 0.8, 0.1]).numpy()
    threshold4 = 0.6
    value4 = 0.0
    input_dict4 = {
        "input": input4,
        "threshold": threshold4,
        "value": value4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(3, 3, 3).numpy()
    threshold5 = -1.0
    value5 = 2.5
    input_dict5 = {
        "input": input5,
        "threshold": threshold5,
        "value": value5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    threshold6 = 2.5
    value6 = 7.0
    input_dict6 = {
        "input": input6,
        "threshold": threshold6,
        "value": value6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([[1.0], [2.0], [3.0]]).numpy()
    threshold7 = 1.5
    value7 = 8.0
    input_dict7 = {
        "input": input7,
        "threshold": threshold7,
        "value": value7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 4, 2).numpy()
    threshold8 = 0.0
    value8 = -5.0
    input_dict8 = {
        "input": input8,
        "threshold": threshold8,
        "value": value8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    threshold9 = -2.5
    value9 = 1.0
    input_dict9 = {
        "input": input9,
        "threshold": threshold9,
        "value": value9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.tensor([1.1, 2.2, 3.3, 4.4]).numpy()
    threshold10 = 3.0
    value10 = 9.9
    input_dict10 = {
        "input": input10,
        "threshold": threshold10,
        "value": value10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.threshold"] = threshold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.threshold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.threshold'.")


check_valid('torch.threshold', generated_inputs['torch.threshold'], lib="torch", suffix=0)
