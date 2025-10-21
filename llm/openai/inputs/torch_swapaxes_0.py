
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def swapaxes_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 4).numpy()
    axis0_1 = 0
    axis1_1 = 1
    input_dict1 = {"input": input1, "axis0": axis0_1, "axis1": axis1_1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(3, 2).numpy()
    axis0_2 = 0
    axis1_2 = 1
    input_dict2 = {"input": input2, "axis0": axis0_2, "axis1": axis1_2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(4, 5, 6, 7).numpy()
    axis0_3 = 1
    axis1_3 = 3
    input_dict3 = {"input": input3, "axis0": axis0_3, "axis1": axis1_3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2, 2, 2).numpy()
    axis0_4 = -1
    axis1_4 = -2
    input_dict4 = {"input": input4, "axis0": axis0_4, "axis1": axis1_4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(5).numpy()
    axis0_5 = 0
    axis1_5 = 0
    input_dict5 = {"input": input5, "axis0": axis0_5, "axis1": axis1_5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 3, 4).numpy()
    axis0_6 = 0
    axis1_6 = 2
    input_dict6 = {"input": input6, "axis0": axis0_6, "axis1": axis1_6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(3, 4, 5).numpy()
    axis0_7 = -3
    axis1_7 = -1
    input_dict7 = {"input": input7, "axis0": axis0_7, "axis1": axis1_7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 3).numpy()
    axis0_8 = 1
    axis1_8 = 0
    input_dict8 = {"input": input8, "axis0": axis0_8, "axis1": axis1_8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(4, 2, 3, 1).numpy()
    axis0_9 = 0
    axis1_9 = 3
    input_dict9 = {"input": input9, "axis0": axis0_9, "axis1": axis1_9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(2, 2, 2).numpy()
    axis0_10 = -1
    axis1_10 = -3
    input_dict10 = {"input": input10, "axis0": axis0_10, "axis1": axis1_10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.swapaxes"] = swapaxes_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.swapaxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.swapaxes'.")


check_valid('torch.swapaxes', generated_inputs['torch.swapaxes'], lib="torch", suffix=0)
