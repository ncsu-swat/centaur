
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def flatten_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    start_dim1 = 1
    end_dim1 = 2
    input_dict1 = {
        "start_dim": start_dim1,
        "end_dim": end_dim1,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.randint(0, 10, size=(5, 2)).astype(np.int64)
    start_dim2 = 0
    end_dim2 = 1
    input_dict2 = {
        "start_dim": start_dim2,
        "end_dim": end_dim2,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(1, 1, 1, 1).astype(np.float32)
    start_dim3 = 0
    end_dim3 = 2
    input_dict3 = {
        "start_dim": start_dim3,
        "end_dim": end_dim3,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(4, 5, 6, 7).astype(np.float32)
    start_dim4 = 1
    end_dim4 = 3
    input_dict4 = {
        "start_dim": start_dim4,
        "end_dim": end_dim4,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.randint(0, 5, size=(2, 2, 2, 2, 2)).astype(np.int32)
    start_dim5 = 2
    end_dim5 = 4
    input_dict5 = {
        "start_dim": start_dim5,
        "end_dim": end_dim5,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(3, 3).astype(np.float64)
    start_dim6 = 0
    end_dim6 = 1
    input_dict6 = {
        "start_dim": start_dim6,
        "end_dim": end_dim6,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(6, 7, 8).astype(np.float16)
    start_dim7 = 1
    end_dim7 = 2
    input_dict7 = {
        "start_dim": start_dim7,
        "end_dim": end_dim7,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.Flatten"] = flatten_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Flatten' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Flatten'.")


check_valid('torch.nn.Flatten', generated_inputs['torch.nn.Flatten'], lib="torch", suffix=0)
