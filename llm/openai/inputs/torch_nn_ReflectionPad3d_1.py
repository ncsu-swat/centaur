
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def reflectionpad3d_inputs():
    list_of_inputs = []
    
    input1 = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2)
    padding1 = 1
    
    input_dict1 = {
        "padding": padding1,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(1, 3, 4, 5, 6).astype(np.float32)
    padding2 = (1, 2, 1, 2, 1, 2)
    
    input_dict2 = {
        "padding": padding2,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.zeros((2, 2, 2, 2, 2), dtype=np.float32)
    padding3 = 0
    
    input_dict3 = {
        "padding": padding3,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    padding4 = (0, 0, 0, 0, 0, 0)
    
    input_dict4 = {
        "padding": padding4,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.rand(4, 3, 2, 2, 2).astype(np.float32)
    padding5 = 1
    
    input_dict5 = {
        "padding": padding5,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(2, 1, 3, 4, 5).astype(np.float32)
    padding6 = (0, 0, 0, 0, 0, 0)
    
    input_dict6 = {
        "padding": padding6,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.nn.ReflectionPad3d_1"] = reflectionpad3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReflectionPad3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReflectionPad3d_1'.")


check_valid('torch.nn.ReflectionPad3d', generated_inputs['torch.nn.ReflectionPad3d_1'], lib="torch", suffix=1)
