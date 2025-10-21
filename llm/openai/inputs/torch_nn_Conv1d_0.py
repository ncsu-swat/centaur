
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def conv1d_inputs():
    list_of_inputs = []
    input_dict_1 = {
        'in_channels': 16,
        'out_channels': 33,
        'kernel_size': 3,
        'stride': 2,
        'padding': 0,
        'dilation': 1,
        'groups': 1,
        'bias': True,
        'padding_mode': 'zeros',
        'dtype': np.float32,
        'input': torch.randn(20, 16, 50)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    return list_of_inputs

generated_inputs["torch.nn.Conv1d"] = conv1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Conv1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Conv1d'.")


check_valid('torch.nn.Conv1d', generated_inputs['torch.nn.Conv1d'], lib="torch", suffix=0)
