
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def full_like_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3])
    fill_value1 = 5.0
    dtype1 = np.float32
    requires_grad1 = False
    
    input_dict1 = {
        "input": input1,
        "fill_value": fill_value1,
        "dtype": torch.float32,
        "requires_grad": requires_grad1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.zeros((2, 2))
    fill_value2 = -1.0
    dtype2 = np.int64
    requires_grad2 = False
    
    input_dict2 = {
        "input": input2,
        "fill_value": fill_value2,
        "dtype": torch.int64,
        "requires_grad": requires_grad2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.ones((3, 4, 5))
    fill_value3 = 0.0
    dtype3 = np.float64
    requires_grad3 = False
    
    input_dict3 = {
        "input": input3,
        "fill_value": fill_value3,
        "dtype": torch.float64,
        "requires_grad": requires_grad3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[1.5, 2.5], [3.5, 4.5]])
    fill_value4 = 10.0
    dtype4 = np.float16
    requires_grad4 = False
    
    input_dict4 = {
        "input": input4,
        "fill_value": fill_value4,
        "dtype": torch.float16,
        "requires_grad": requires_grad4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1])
    fill_value5 = -5.5
    dtype5 = np.float64
    requires_grad5 = False
    
    input_dict5 = {
        "input": input5,
        "fill_value": fill_value5,
        "dtype": torch.float64,
        "requires_grad": requires_grad5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.full_like"] = full_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.full_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.full_like'.")


check_valid('torch.full_like', generated_inputs['torch.full_like'], lib="torch", suffix=0)
