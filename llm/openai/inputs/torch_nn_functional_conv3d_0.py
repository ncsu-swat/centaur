
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def conv3d_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(20, 16, 50, 10, 20).astype(np.float32)
    weight1 = np.random.rand(33, 16, 3, 3, 3).astype(np.float32)
    bias1 = np.random.rand(33).astype(np.float32)
    stride1 = (1, 1, 1)
    padding1 = 'valid'
    dilation1 = (1, 1, 1)
    groups1 = 1
    
    input_dict1 = {
        "input": input1,
        "weight": weight1,
        "bias": bias1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "groups": groups1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(10, 32, 25, 5, 10).astype(np.float32)
    weight2 = np.random.rand(64, 32, 3, 3, 3).astype(np.float32)
    bias2 = np.random.rand(64).astype(np.float32)
    stride2 = (1, 1, 1)
    padding2 = 'valid'
    dilation2 = (1, 1, 1)
    groups2 = 1
    
    input_dict2 = {
        "input": input2,
        "weight": weight2,
        "bias": bias2,
        "stride": stride2,
        "padding": padding2,
        "dilation": dilation2,
        "groups": groups2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(5, 8, 16, 8, 8).astype(np.float32)
    weight3 = np.random.rand(16, 8, 3, 3, 3).astype(np.float32)
    bias3 = np.random.rand(16).astype(np.float32)
    stride3 = (1, 1, 1)
    padding3 = 1
    dilation3 = (1, 1, 1)
    groups3 = 1
    
    input_dict3 = {
        "input": input3,
        "weight": weight3,
        "bias": bias3,
        "stride": stride3,
        "padding": padding3,
        "dilation": dilation3,
        "groups": groups3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(4, 32, 16, 16, 16).astype(np.float32)
    weight4 = np.random.rand(64, 32, 3, 3, 3).astype(np.float32)
    bias4 = np.random.rand(64).astype(np.float32)
    stride4 = (1, 1, 1)
    padding4 = 0
    dilation4 = (1, 1, 1)
    groups4 = 1
    
    input_dict4 = {
        "input": input4,
        "weight": weight4,
        "bias": bias4,
        "stride": stride4,
        "padding": padding4,
        "dilation": dilation4,
        "groups": groups4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    
    return list_of_inputs

generated_inputs["torch.nn.functional.conv3d"] = conv3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.conv3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv3d'.")


check_valid('torch.nn.functional.conv3d', generated_inputs['torch.nn.functional.conv3d'], lib="torch", suffix=0)
