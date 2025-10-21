
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def randn_like_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    dtype1 = torch.float32
    requires_grad1 = True
    input_dict1 = {
        "input": input1,
        "dtype": dtype1,
        "requires_grad": requires_grad1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1, 2], [3, 4]])
    dtype2 = torch.float64
    requires_grad2 = False
    input_dict2 = {
        "input": input2,
        "dtype": dtype2,
        "requires_grad": requires_grad2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.zeros((2, 3, 4))
    dtype3 = torch.float16
    requires_grad3 = True
    input_dict3 = {
        "input": input3,
        "dtype": dtype3,
        "requires_grad": requires_grad3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(5)
    dtype4 = torch.float32
    requires_grad4 = False
    input_dict4 = {
        "input": input4,
        "dtype": dtype4,
        "requires_grad": requires_grad4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[-1, 2], [3, -4]])
    dtype5 = torch.float64
    requires_grad5 = True
    input_dict5 = {
        "input": input5,
        "dtype": dtype5,
        "requires_grad": requires_grad5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.randn_like"] = randn_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.randn_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.randn_like'.")


check_valid('torch.randn_like', generated_inputs['torch.randn_like'], lib="torch", suffix=0)
