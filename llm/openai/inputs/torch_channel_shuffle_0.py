
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def channel_shuffle_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(4, 3, 32, 32).astype(np.float32)
    groups1 = 1
    input_dict1 = {"input": input1, "groups": groups1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(1, 64, 64, 64).astype(np.float32)
    groups2 = 4
    input_dict2 = {"input": input2, "groups": groups2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(8, 128, 16, 16).astype(np.float32)
    groups3 = 8
    input_dict3 = {"input": input3, "groups": groups3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 6, 10, 10).astype(np.float32)
    groups4 = 3
    input_dict4 = {"input": input4, "groups": groups4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(16, 32, 8, 8).astype(np.float32)
    groups5 = 16
    input_dict5 = {"input": input5, "groups": groups5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(4, 64, 128, 128).astype(np.float32)
    groups6 = 32
    input_dict6 = {"input": input6, "groups": groups6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(1, 3, 256, 256).astype(np.float32)
    groups7 = 1
    input_dict7 = {"input": input7, "groups": groups7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(32, 4, 32, 32).astype(np.float32)
    groups8 = 2
    input_dict8 = {"input": input8, "groups": groups8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(2, 8, 64, 64).astype(np.float32)
    groups9 = 4
    input_dict9 = {"input": input9, "groups": groups9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    return list_of_inputs

generated_inputs["torch.channel_shuffle"] = channel_shuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.channel_shuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.channel_shuffle'.")


check_valid('torch.channel_shuffle', generated_inputs['torch.channel_shuffle'], lib="torch", suffix=0)
