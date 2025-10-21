
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def linear_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3).astype(np.float32)
    weight1 = np.random.rand(4, 3).astype(np.float32)
    bias1 = np.random.rand(4).astype(np.float32)
    
    input_dict1 = {
        "input": input1,
        "weight": weight1,
        "bias": bias1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(5, 7).astype(np.float32)
    weight2 = np.random.rand(3, 7).astype(np.float32)
    bias2 = np.random.rand(3).astype(np.float32)
    
    input_dict2 = {
        "input": input2,
        "weight": weight2,
        "bias": bias2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(1, 10).astype(np.float32)
    weight3 = np.random.rand(8, 10).astype(np.float32)
    bias3 = np.random.rand(8).astype(np.float32)

    input_dict3 = {
        "input": input3,
        "weight": weight3,
        "bias": bias3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 3).astype(np.float32)
    weight4 = np.random.rand(5, 3).astype(np.float32)
    bias4 = np.random.rand(5).astype(np.float32)

    input_dict4 = {
        "input": input4,
        "weight": weight4,
        "bias": bias4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(3, 4).astype(np.float32)
    weight5 = np.random.rand(7, 4).astype(np.float32)
    bias5 = np.random.rand(7).astype(np.float32)

    input_dict5 = {
        "input": input5,
        "weight": weight5,
        "bias": bias5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(4, 1).astype(np.float32)
    weight6 = np.random.rand(2, 1).astype(np.float32)
    bias6 = np.random.rand(2).astype(np.float32)

    input_dict6 = {
        "input": input6,
        "weight": weight6,
        "bias": bias6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(2, 3).astype(np.float32)
    weight7 = np.random.rand(2, 3).astype(np.float32)
    bias7 = np.random.rand(2).astype(np.float32)

    input_dict7 = {
        "input": input7,
        "weight": weight7,
        "bias": bias7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(5).astype(np.float32).reshape(1, -1)
    weight8 = np.random.rand(3, 5).astype(np.float32)
    bias8 = np.random.rand(3).astype(np.float32)

    input_dict8 = {
        "input": input8,
        "weight": weight8,
        "bias": bias8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.linear"] = linear_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.linear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.linear'.")


check_valid('torch.nn.functional.linear', generated_inputs['torch.nn.functional.linear'], lib="torch", suffix=0)
