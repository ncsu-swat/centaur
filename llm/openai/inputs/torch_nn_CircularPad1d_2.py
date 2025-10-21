
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def circular_pad_1d_inputs():
    list_of_inputs = []

    input1 = torch.arange(8, dtype=torch.float).reshape(1, 2, 4).numpy()
    padding1 = 2
    input_dict1 = {"padding": [padding1, padding1], "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 3, 5).numpy()
    padding2 = (1, 2)
    input_dict2 = {"padding": padding2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.nn.CircularPad1d_2"] = circular_pad_1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.CircularPad1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CircularPad1d_2'.")


check_valid('torch.nn.CircularPad1d', generated_inputs['torch.nn.CircularPad1d_2'], lib="torch", suffix=2)
