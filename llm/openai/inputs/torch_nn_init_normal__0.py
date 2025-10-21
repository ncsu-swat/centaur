
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def normal_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3).astype(np.float32)
    mean1 = 0.0
    std1 = 1.0
    input_dict1 = {
        "tensor": input1,
        "mean": mean1,
        "std": std1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.zeros((4, 4, 4)).astype(np.float64)
    mean2 = 2.5
    std2 = 0.5
    input_dict2 = {
        "tensor": input2,
        "mean": mean2,
        "std": std2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(5).astype(np.float16)
    mean3 = -1.0
    std3 = 2.0
    input_dict3 = {
        "tensor": input3,
        "mean": mean3,
        "std": std3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.ones((1, 2, 3, 4)).astype(np.float32)
    mean4 = 1.5
    std4 = 0.25
    input_dict4 = {
        "tensor": input4,
        "mean": mean4,
        "std": std4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.rand(3, 3).astype(np.float64)
    mean5 = -0.5
    std5 = 1.5
    input_dict5 = {
        "tensor": input5,
        "mean": mean5,
        "std": std5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    mean6 = 0.0
    std6 = 0.1
    input_dict6 = {
        "tensor": input6,
        "mean": mean6,
        "std": std6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.rand(2, 2, 2).astype(np.float64)
    mean7 = 5.0
    std7 = 3.0
    input_dict7 = {
        "tensor": input7,
        "mean": mean7,
        "std": std7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.zeros((10,)).astype(np.float32)
    input8[:] = 7.0
    mean8 = 7.0
    std8 = 0.01
    input_dict8 = {
        "tensor": input8,
        "mean": mean8,
        "std": std8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.random.rand(2, 3, 4, 5).astype(np.float16)
    mean9 = -2.0
    std9 = 0.75
    input_dict9 = {
        "tensor": input9,
        "mean": mean9,
        "std": std9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.zeros((1,)).astype(np.float32)
    mean10 = 10.0
    std10 = 1.0
    input_dict10 = {
        "tensor": input10,
        "mean": mean10,
        "std": std10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.init.normal_"] = normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.normal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.normal_'.")


check_valid('torch.nn.init.normal_', generated_inputs['torch.nn.init.normal_'], lib="torch", suffix=0)
