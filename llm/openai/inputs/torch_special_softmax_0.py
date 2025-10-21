
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def softmax_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dim1 = 0
    dtype1 = torch.float32
    
    input_dict = {
        "input": input1,
        "dim": dim1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input2 = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    dim2 = 0
    dtype2 = torch.float64
    
    input_dict = {
        "input": input2,
        "dim": dim2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dim3 = 1
    dtype3 = torch.float32
    
    input_dict = {
        "input": input3,
        "dim": dim3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.special.softmax"] = softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.softmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.softmax'.")


check_valid('torch.special.softmax', generated_inputs['torch.special.softmax'], lib="torch", suffix=0)
