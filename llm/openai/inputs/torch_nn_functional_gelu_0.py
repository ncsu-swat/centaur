
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def gelu_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    approximate1 = "none"
    
    input_dict1 = {
        "input": input1,
        "approximate": approximate1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    approximate2 = "tanh"
    
    input_dict2 = {
        "input": input2,
        "approximate": approximate2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(2, 2).astype(np.float32)
    approximate3 = "none"
    
    input_dict3 = {
        "input": input3,
        "approximate": approximate3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.randn(3, 3, 3).astype(np.float32)
    approximate4 = "tanh"
    
    input_dict4 = {
        "input": input4,
        "approximate": approximate4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([0.0], dtype=np.float32)
    approximate5 = "none"
    
    input_dict5 = {
        "input": input5,
        "approximate": approximate5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([1.5, -2.5, 3.5], dtype=np.float32)
    approximate6 = "tanh"
    
    input_dict6 = {
        "input": input6,
        "approximate": approximate6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.full((4, 1), 2.0, dtype=np.float32)
    approximate7 = "none"

    input_dict7 = {
        "input": input7,
        "approximate": approximate7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.zeros((2, 4), dtype=np.float32)
    approximate8 = "tanh"

    input_dict8 = {
        "input": input8,
        "approximate": approximate8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 2.0], dtype=np.float64)
    approximate9 = "none"
    
    input_dict9 = {
        "input": input9,
        "approximate": approximate9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([-1.0, -2.0], dtype=np.float64)
    approximate10 = "tanh"
    
    input_dict10 = {
        "input": input10,
        "approximate": approximate10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.gelu"] = gelu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.gelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.gelu'.")


check_valid('torch.nn.functional.gelu', generated_inputs['torch.nn.functional.gelu'], lib="torch", suffix=0)
