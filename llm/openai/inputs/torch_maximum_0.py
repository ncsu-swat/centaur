
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def maximum_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3])
    other1 = np.array([4, 5, 6])
    out1 = np.array([0, 0, 0], dtype=np.int64)
    
    input_dict1 = {
        "input": input1,
        "other": other1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([-1, -2, -3])
    other2 = np.array([1, 2, 3])
    out2 = np.array([0, 0, 0], dtype=np.int64)
    
    input_dict2 = {
        "input": input2,
        "other": other2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1, 2], [3, 4]])
    other3 = np.array([[5, 6], [7, 8]])
    out3 = np.array([[0, 0], [0, 0]], dtype=np.int64)
    
    input_dict3 = {
        "input": input3,
        "other": other3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.maximum"] = maximum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.maximum'.")


check_valid('torch.maximum', generated_inputs['torch.maximum'], lib="torch", suffix=0)
