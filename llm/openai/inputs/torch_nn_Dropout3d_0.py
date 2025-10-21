
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def dropout3d_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(20, 16, 4, 32, 32).astype(np.float32)
    input_dict1 = {"p": 0.2, "inplace": False, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(10, 8, 2, 16, 16).astype(np.float32)
    input_dict2 = {"p": 0.5, "inplace": True, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(5, 32, 8, 8, 8).astype(np.float32)
    input_dict3 = {"p": 0.0, "inplace": False, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(1, 1, 1, 1, 1).astype(np.float32)
    input_dict4 = {"p": 1.0, "inplace": True, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.rand(32, 16, 4, 32, 32).astype(np.float32)
    input_dict5 = {"p": 0.7, "inplace": False, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(8, 128, 1, 32, 32).astype(np.float32)
    input_dict6 = {"p": 0.3, "inplace": False, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.rand(16, 64, 5, 16, 16).astype(np.float32)
    input_dict7 = {"p": 0.9, "inplace": True, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(2, 2, 2, 2, 2).astype(np.float32)
    input_dict8 = {"p": 0.4, "inplace": False, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.random.rand(64, 3, 8, 16, 32).astype(np.float32)
    input_dict9 = {"p": 0.6, "inplace": True, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["torch.nn.Dropout3d"] = dropout3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Dropout3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Dropout3d'.")


check_valid('torch.nn.Dropout3d', generated_inputs['torch.nn.Dropout3d'], lib="torch", suffix=0)
