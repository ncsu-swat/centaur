
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cumsum_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3, 4, 5])
    dim1 = 0
    dtype1 = np.float32
    
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 2, 3], [4, 5, 6]])
    dim2 = 1
    dtype2 = np.int64
    
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1, 2, 3, 4, 5])
    dim3 = 0
    dtype3 = None
    out3 = np.zeros(5)
    
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "dtype": dtype3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.cumsum"] = cumsum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cumsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cumsum'.")


check_valid('torch.cumsum', generated_inputs['torch.cumsum'], lib="torch", suffix=0)
