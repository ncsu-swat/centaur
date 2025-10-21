
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def init_zeros_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3).numpy()
    input_dict1 = {"tensor": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.zeros(5).numpy()
    input_dict2 = {"tensor": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.ones((2, 2, 2)).numpy()
    input_dict3 = {"tensor": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randint(0, 10, (3, 4)).numpy()
    input_dict4 = {"tensor": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 2, 3, 4).numpy()
    input_dict5 = {"tensor": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.empty(2, 3).numpy()
    input_dict6 = {"tensor": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict7 = {"tensor": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.arange(10).numpy()
    input_dict8 = {"tensor": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(2, 2).numpy()
    input_dict9 = {"tensor": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.tensor([[1.5, 2.5], [3.5, 4.5]]).numpy()
    input_dict10 = {"tensor": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    input11 = torch.randn(3, 3, 3).numpy()
    input_dict11 = {"tensor": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs["torch.nn.init.zeros_"] = init_zeros_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.zeros_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.zeros_'.")


check_valid('torch.nn.init.zeros_', generated_inputs['torch.nn.init.zeros_'], lib="torch", suffix=0)
