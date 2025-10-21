
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ldexp_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    other1 = np.array([0, 1, 2], dtype=np.int64)
    
    input_dict1 = {
        "input": input1,
        "other": other1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[-1.0, 2.5], [3.0, -4.5]], dtype=np.float32)
    other2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    
    input_dict2 = {
        "input": input2,
        "other": other2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.0, -1.0, 2.0], dtype=np.float16)
    other3 = np.array([5, -2, 10], dtype=np.int16)
    
    input_dict3 = {
        "input": input3,
        "other": other3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    other4 = np.array([10], dtype=np.int64)
    
    input_dict4 = {
        "input": input4,
        "other": other4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    other5 = np.array([1], dtype=np.int32)
    
    input_dict5 = {
        "input": input5,
        "other": other5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    other6 = np.array([-1], dtype=np.int64)
    
    input_dict6 = {
        "input": input6,
        "other": other6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    other7 = np.array([0], dtype=np.int32)

    input_dict7 = {
        "input": input7,
        "other": other7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.ldexp_"] = ldexp_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ldexp_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ldexp_'.")


check_valid('torch.ldexp_', generated_inputs['torch.ldexp_'], lib="torch", suffix=0)
