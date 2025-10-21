
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def zeropad2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 1, 3, 3).numpy()
    padding1 = (1, 1, 1, 1)
    input_dict1 = {"padding": padding1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 3, 5, 5).numpy()
    padding2 = (2, 2, 2, 2)
    input_dict2 = {"padding": padding2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 4, 4).numpy()
    padding3 = (0, 2, 1, 0)
    input_dict3 = {"padding": padding3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(3, 1, 2, 2).numpy()
    padding4 = (3, 3, 3, 3)
    input_dict4 = {"padding": padding4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 1, 1, 1).numpy()
    padding5 = (0, 0, 0, 0)
    input_dict5 = {"padding": padding5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 1, 3, 3).numpy()
    padding6 = (1, 2, 3, 4)
    input_dict6 = {"padding": padding6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.nn.ZeroPad2d_2"] = zeropad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ZeroPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ZeroPad2d_2'.")


check_valid('torch.nn.ZeroPad2d', generated_inputs['torch.nn.ZeroPad2d_2'], lib="torch", suffix=2)
