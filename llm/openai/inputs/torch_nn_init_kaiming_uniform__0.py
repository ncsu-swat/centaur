
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def kaiming_uniform_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3).astype(np.float32)
    a1 = 1.0
    mode1 = 'fan_in'
    nonlinearity1 = 'leaky_relu'
    input_dict1 = {
        "tensor": input1,
        "a": a1,
        "mode": mode1,
        "nonlinearity": nonlinearity1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(5, 5, 5).astype(np.float64)
    a2 = 0.5
    mode2 = 'fan_out'
    nonlinearity2 = 'relu'
    input_dict2 = {
        "tensor": input2,
        "a": a2,
        "mode": mode2,
        "nonlinearity": nonlinearity2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(10, 2).astype(np.float32)
    a3 = 2.0
    mode3 = 'fan_in'
    nonlinearity3 = 'sigmoid'
    input_dict3 = {
        "tensor": input3,
        "a": a3,
        "mode": mode3,
        "nonlinearity": nonlinearity3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 2).astype(np.float16)
    a4 = 0.1
    mode4 = 'fan_out'
    nonlinearity4 = 'tanh'
    input_dict4 = {
        "tensor": input4,
        "a": a4,
        "mode": mode4,
        "nonlinearity": nonlinearity4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(3, 4, 5).astype(np.float32)
    a5 = 1.5
    mode5 = 'fan_in'
    nonlinearity5 = 'leaky_relu'
    input_dict5 = {
        "tensor": input5,
        "a": a5,
        "mode": mode5,
        "nonlinearity": nonlinearity5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(7, 7).astype(np.float64)
    a6 = -0.5
    mode6 = 'fan_out'
    nonlinearity6 = 'relu'
    input_dict6 = {
        "tensor": input6,
        "a": a6,
        "mode": mode6,
        "nonlinearity": nonlinearity6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.nn.init.kaiming_uniform_"] = kaiming_uniform_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.kaiming_uniform_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.kaiming_uniform_'.")


check_valid('torch.nn.init.kaiming_uniform_', generated_inputs['torch.nn.init.kaiming_uniform_'], lib="torch", suffix=0)
