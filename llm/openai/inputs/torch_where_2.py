
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_where_inputs():
    list_of_inputs = []
    
    condition = np.array([True, False, True, False])
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0])
    other_tensor = np.array([5.0, 6.0, 7.0, 8.0])
    out_tensor = np.empty_like(condition, dtype=np.float64)
    list_of_inputs.append({
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": out_tensor
    })
    
    condition = np.array([[True, False], [False, True]])
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    other_tensor = np.array([[5.0, 6.0], [7.0, 8.0]])
    out_tensor = np.empty_like(condition, dtype=np.float64)
    list_of_inputs.append({
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": out_tensor
    })

    condition = np.array([True, False, True])
    input_tensor = np.array([1.0, 2.0, 3.0])
    other_tensor = np.array([5.0, 6.0, 7.0])
    out_tensor = np.empty_like(condition, dtype=np.float64)
    list_of_inputs.append({
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": out_tensor
    })

    condition = np.array([[False, True], [True, False]])
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    other_tensor = np.array([[5.0, 6.0], [7.0, 8.0]])
    out_tensor = np.empty_like(condition, dtype=np.float64)
    list_of_inputs.append({
        "condition": condition,
        "input": input_tensor,
        "other": other_tensor,
        "out": out_tensor
    })
    
    return list_of_inputs

generated_inputs["torch.where_2"] = torch_where_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.where_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.where_2'.")


check_valid('torch.where', generated_inputs['torch.where_2'], lib="torch", suffix=2)
