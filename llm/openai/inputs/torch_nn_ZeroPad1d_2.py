
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def zeropad1d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 2, 4).numpy()
    padding1 = (2, 2)
    input_dict1 = {"input": input1, "padding": padding1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 2, 3).numpy()
    padding2 = (3, 1)
    input_dict2 = {"input": input2, "padding": padding2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 5).numpy()
    padding3 = (1, 1)
    input_dict3 = {"input": input3, "padding": padding3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 4, 6).numpy()
    padding4 = (1, 2)
    input_dict4 = {"input": input4, "padding": padding4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 3, 7).numpy()
    padding5 = (0, 0)
    input_dict5 = {"input": input5, "padding": padding5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 1, 5).numpy()
    padding6 = (2, 2)
    input_dict6 = {"input": input6, "padding": padding6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(4, 2).numpy()
    padding7 = (1, 1)
    input_dict7 = {"input": input7, "padding": padding7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 1, 3).numpy()
    padding8 = (0, 1)
    input_dict8 = {"input": input8, "padding": padding8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs["torch.nn.ZeroPad1d_2"] = zeropad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ZeroPad1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ZeroPad1d_2'.")


check_valid('torch.nn.ZeroPad1d', generated_inputs['torch.nn.ZeroPad1d_2'], lib="torch", suffix=2)
