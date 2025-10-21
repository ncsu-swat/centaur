
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bilinear_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    input2 = np.random.rand(2, 3, 5).astype(np.float32)
    weight = np.random.rand(6, 4, 5).astype(np.float32)
    bias = np.random.rand(6).astype(np.float32)
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.random.rand(1, 2, 3).astype(np.float32)
    input2 = np.random.rand(1, 2, 4).astype(np.float32)
    weight = np.random.rand(5, 3, 4).astype(np.float32)
    bias = np.random.rand(5).astype(np.float32)
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.random.rand(3, 2, 2).astype(np.float32)
    input2 = np.random.rand(3, 2, 3).astype(np.float32)
    weight = np.random.rand(4, 2, 3).astype(np.float32)
    bias = np.random.rand(4).astype(np.float32)
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.random.rand(4, 5, 6).astype(np.float32)
    input2 = np.random.rand(4, 5, 7).astype(np.float32)
    weight = np.random.rand(8, 6, 7).astype(np.float32)
    bias = np.random.rand(8).astype(np.float32)
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    input2 = np.random.rand(2, 3, 5).astype(np.float32)
    weight = np.random.rand(6, 4, 5).astype(np.float32)
    bias = np.zeros(6).astype(np.float32)
    
    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.rand(1, 1, 2).astype(np.float32)
    input2 = np.random.rand(1, 1, 3).astype(np.float32)
    weight = np.random.rand(2, 2, 3).astype(np.float32)
    bias = np.random.rand(2).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.random.rand(5, 4, 3).astype(np.float32)
    input2 = np.random.rand(5, 4, 2).astype(np.float32)
    weight = np.random.rand(7, 3, 2).astype(np.float32)
    bias = np.random.rand(7).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    input2 = np.random.rand(2, 3, 5).astype(np.float32)
    weight = np.random.rand(6, 4, 5).astype(np.float32)
    bias = np.random.rand(6).astype(np.float32) * -1

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.random.rand(10, 2, 5).astype(np.float32)
    input2 = np.random.rand(10, 2, 3).astype(np.float32)
    weight = np.random.rand(8, 5, 3).astype(np.float32)
    bias = np.random.rand(8).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.random.rand(2, 2, 2).astype(np.float32)
    input2 = np.random.rand(2, 2, 2).astype(np.float32)
    weight = np.random.rand(3, 2, 2).astype(np.float32)
    bias = np.random.rand(3).astype(np.float32)

    input_dict = {
        "input1": input1,
        "input2": input2,
        "weight": weight,
        "bias": bias
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.bilinear"] = bilinear_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.bilinear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.bilinear'.")


check_valid('torch.nn.functional.bilinear', generated_inputs['torch.nn.functional.bilinear'], lib="torch", suffix=0)
