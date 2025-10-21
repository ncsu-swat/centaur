
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def unflatten_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
    dim1 = 0
    unflattened_size1 = (2, 3)
    
    input_dict1 = {
        "dim": dim1,
        "unflattened_size": unflattened_size1,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.int64)
    dim2 = 0
    unflattened_size2 = (3, 3)
    
    input_dict2 = {
        "dim": dim2,
        "unflattened_size": unflattened_size2,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([1, 2, 3, 4]).astype(np.float64)
    dim3 = 0
    unflattened_size3 = (2, 2)
    
    input_dict3 = {
        "dim": dim3,
        "unflattened_size": unflattened_size3,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).astype(np.int32)
    dim4 = 0
    unflattened_size4 = (4, 3)
    
    input_dict4 = {
        "dim": dim4,
        "unflattened_size": unflattened_size4,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1, 2, 3]).astype(np.float16)
    dim5 = 0
    unflattened_size5 = (1, 3)
    
    input_dict5 = {
        "dim": dim5,
        "unflattened_size": unflattened_size5,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.int64)
    dim6 = 0
    unflattened_size6 = (2, 4)

    input_dict6 = {
        "dim": dim6,
        "unflattened_size": unflattened_size6,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.nn.Unflatten"] = unflatten_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Unflatten' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Unflatten'.")


check_valid('torch.nn.Unflatten', generated_inputs['torch.nn.Unflatten'], lib="torch", suffix=0)
