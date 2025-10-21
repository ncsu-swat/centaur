
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_where_inputs():
    list_of_inputs = []
    
    condition = np.array([True, False, True, False], dtype=bool)
    input_tensor = np.array([1, 2, 3, 4], dtype=np.int64)
    other_tensor = np.array([5, 6, 7, 8], dtype=np.int64)
    out_tensor = np.zeros(4, dtype=np.int64)
    
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([[True, False], [False, True]], dtype=bool)
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other_tensor = np.array([[5, 6], [7, 8]], dtype=np.int64)
    out_tensor = np.zeros((2, 2), dtype=np.int64)
    
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    condition = np.array([False, True, False, True], dtype=bool)
    input_tensor = np.array([-1, -2, -3, -4], dtype=np.int64)
    other_tensor = np.array([10, 20, 30, 40], dtype=np.int64)
    out_tensor = np.zeros(4, dtype=np.int64)
    
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([True, True, True], dtype=bool)
    input_tensor = np.array([1, 2, 3], dtype=np.int64)
    other_tensor = np.array([4, 5, 6], dtype=np.int64)
    out_tensor = np.zeros(3, dtype=np.int64)
    
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    condition = np.array([[False, False], [True, True]], dtype=bool)
    input_tensor = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other_tensor = np.array([[5, 6], [7, 8]], dtype=np.int64)
    out_tensor = np.zeros((2, 2), dtype=np.int64)
    
    input_dict = {
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.where_1"] = torch_where_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.where_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.where_1'.")


check_valid('torch.where', generated_inputs['torch.where_1'], lib="torch", suffix=1)
