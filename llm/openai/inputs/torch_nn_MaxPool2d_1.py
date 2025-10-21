
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def maxpool2d_inputs():
    list_of_inputs = []
    input_dict = {}

    input_1 = torch.randn(20, 16, 50, 32).numpy()
    input_dict['input'] = input_1
    input_dict['kernel_size'] = 3
    input_dict['stride'] = 2
    input_dict['padding'] = 0
    input_dict['dilation'] = 1
    input_dict['return_indices'] = False
    input_dict['ceil_mode'] = False
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    input_2 = torch.randn(1, 3, 64, 64).numpy()
    input_dict['input'] = input_2
    input_dict['kernel_size'] = (3, 2)
    input_dict['stride'] = (2, 1)
    input_dict['padding'] = 1
    input_dict['dilation'] = 1
    input_dict['return_indices'] = True
    input_dict['ceil_mode'] = False
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    input_3 = torch.randn(10, 1, 32, 32).numpy()
    input_dict['input'] = input_3
    input_dict['kernel_size'] = 2
    input_dict['stride'] = 1
    input_dict['padding'] = 0
    input_dict['dilation'] = 2
    input_dict['return_indices'] = False
    input_dict['ceil_mode'] = True
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    input_4 = torch.randn(5, 8, 128, 128).numpy()
    input_dict['input'] = input_4
    input_dict['kernel_size'] = 5
    input_dict['stride'] = 5
    input_dict['padding'] = 0
    input_dict['dilation'] = 1
    input_dict['return_indices'] = True
    input_dict['ceil_mode'] = True
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    input_5 = torch.randn(1, 16, 32, 32).numpy()
    input_dict['input'] = input_5
    input_dict['kernel_size'] = 2
    input_dict['stride'] = 2
    input_dict['padding'] = 0
    input_dict['dilation'] = 1
    input_dict['return_indices'] = False
    input_dict['ceil_mode'] = False
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_1"] = maxpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool2d_1'.")


check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_1'], lib="torch", suffix=1)
