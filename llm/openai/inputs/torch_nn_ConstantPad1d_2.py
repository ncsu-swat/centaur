
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def constant_pad1d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 2, 4).numpy()
    padding1 = (2, 2)
    value1 = 3.5
    input_dict1 = {"input": input1, "padding": padding1, "value": value1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 2, 3).numpy()
    padding2 = (3, 1)
    value2 = 3.5
    input_dict2 = {"input": input2, "padding": padding2, "value": value2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 5).numpy()
    padding3 = (1, 1)
    value3 = 0.0
    input_dict3 = {"input": input3, "padding": padding3, "value": value3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3, 6).numpy()
    padding4 = (1, 2)
    value4 = -1.0
    input_dict4 = {"input": input4, "padding": padding4, "value": value4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.nn.ConstantPad1d_2"] = constant_pad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ConstantPad1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad1d_2'.")


check_valid('torch.nn.ConstantPad1d', generated_inputs['torch.nn.ConstantPad1d_2'], lib="torch", suffix=2)
