
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def diagonal_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(3, 3).astype(np.float32)
    input_dict1 = {"input": input1, "offset": 0, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(2, 5).astype(np.float64)
    input_dict2 = {"input": input2, "offset": 1, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(4, 4).astype(np.float32)
    input_dict3 = {"input": input3, "offset": -1, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(2, 3, 3).astype(np.float64)
    input_dict4 = {"input": input4, "offset": 0, "dim1": 1, "dim2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.rand(2, 5, 4, 2).astype(np.float32)
    input_dict5 = {"input": input5, "offset": -1, "dim1": 1, "dim2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(5, 5).astype(np.float64)
    input_dict6 = {"input": input6, "offset": 2, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(3, 3).astype(np.float32)
    input_dict7 = {"input": input7, "offset": -2, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.random.rand(2, 2).astype(np.float64)
    input_dict8 = {"input": input8, "offset": 0, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(4, 5).astype(np.float32)
    input_dict9 = {"input": input9, "offset": 1, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.random.rand(2, 4, 4).astype(np.float64)
    input_dict10 = {"input": input10, "offset": -1, "dim1": 1, "dim2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.diagonal"] = diagonal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.diagonal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diagonal'.")


check_valid('torch.diagonal', generated_inputs['torch.diagonal'], lib="torch", suffix=0)
