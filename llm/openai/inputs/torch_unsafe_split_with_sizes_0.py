
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def unsafe_split_with_sizes_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(10).astype(np.float32)
    split_sizes1 = [2, 3, 5]
    dim1 = 0
    
    input_dict1 = {
        "input": input1,
        "split_sizes": split_sizes1,
        "dim": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(4, 5).astype(np.float64)
    split_sizes2 = [1, 1, 1, 2]
    dim2 = 1
    
    input_dict2 = {
        "input": input2,
        "split_sizes": split_sizes2,
        "dim": dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    split_sizes3 = [1, 1, 2]
    dim3 = 2
    
    input_dict3 = {
        "input": input3,
        "split_sizes": split_sizes3,
        "dim": dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(5, 5).astype(np.float16)
    split_sizes4 = [2, 3]
    dim4 = 0
    
    input_dict4 = {
        "input": input4,
        "split_sizes": split_sizes4,
        "dim": dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(12).astype(np.float32)
    split_sizes5 = [3, 4, 5]
    dim5 = 0

    input_dict5 = {
        "input": input5,
        "split_sizes": split_sizes5,
        "dim": dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    

    input6 = np.random.rand(2, 4, 6).astype(np.float64)
    split_sizes6 = [1, 1, 4]
    dim6 = 2

    input_dict6 = {
        "input": input6,
        "split_sizes": split_sizes6,
        "dim": dim6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(2, 2).astype(np.float32)
    split_sizes7 = [1, 1]
    dim7 = 0

    input_dict7 = {
        "input": input7,
        "split_sizes": split_sizes7,
        "dim": dim7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.unsafe_split_with_sizes"] = unsafe_split_with_sizes_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.unsafe_split_with_sizes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unsafe_split_with_sizes'.")


check_valid('torch.unsafe_split_with_sizes', generated_inputs['torch.unsafe_split_with_sizes'], lib="torch", suffix=0)
