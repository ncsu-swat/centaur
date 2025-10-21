
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def conv1d_inputs():
    list_of_inputs = []
    input_dict = {}

    input = np.random.rand(33, 16, 30).astype(np.float32)
    weight = np.random.rand(20, 16, 5).astype(np.float32)
    bias = np.random.rand(20).astype(np.float32)
    stride = 1
    padding = 'valid'
    dilation = 1
    groups = 1
    input_dict = {
        'input': input,
        'weight': weight,
        'bias': bias,
        'stride': stride,
        'padding': padding,
        'dilation': dilation,
        'groups': groups
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.rand(1, 1, 10).astype(np.float32)
    weight = np.random.rand(2, 1, 3).astype(np.float32)
    bias = np.random.rand(2).astype(np.float32)
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    input_dict = {
        'input': input,
        'weight': weight,
        'bias': bias,
        'stride': stride,
        'padding': padding,
        'dilation': dilation,
        'groups': groups
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.rand(2, 3, 5).astype(np.float32)
    weight = np.random.rand(4, 3, 2).astype(np.float32)
    bias = np.random.rand(4).astype(np.float32)
    stride = 1
    padding = 'same'
    dilation = 1
    groups = 1
    input_dict = {
        'input': input,
        'weight': weight,
        'bias': bias,
        'stride': stride,
        'padding': padding,
        'dilation': dilation,
        'groups': groups
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.rand(4, 2, 8).astype(np.float32)
    weight = np.random.rand(3, 2, 3).astype(np.float32)
    bias = np.random.rand(3).astype(np.float32)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    input_dict = {
        'input': input,
        'weight': weight,
        'bias': bias,
        'stride': stride,
        'padding': padding,
        'dilation': dilation,
        'groups': groups
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.rand(1, 5, 15).astype(np.float32)
    weight = np.random.rand(8, 5, 4).astype(np.float32)
    bias = np.random.rand(8).astype(np.float32)
    stride = 1
    padding = 2
    dilation = 1
    groups = 1
    input_dict = {
        'input': input,
        'weight': weight,
        'bias': bias,
        'stride': stride,
        'padding': padding,
        'dilation': dilation,
        'groups': groups
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.conv1d"] = conv1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.conv1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv1d'.")


check_valid('torch.nn.functional.conv1d', generated_inputs['torch.nn.functional.conv1d'], lib="torch", suffix=0)
