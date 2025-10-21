
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hardshrink_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(2).numpy()
    lambd1 = 0.5
    input_dict1 = {"lambd": lambd1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(3, 4).numpy()
    lambd2 = 0.1
    input_dict2 = {"lambd": lambd2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(5, 2, 3).numpy()
    lambd3 = 1.0
    input_dict3 = {"lambd": lambd3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([-1.0, -2.0, 3.0]).numpy()
    lambd4 = 0.2
    input_dict4 = {"lambd": lambd4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([0.0, 0.5, 1.0, -0.5, -1.0]).numpy()
    lambd5 = 0.7
    input_dict5 = {"lambd": lambd5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 1, 1, 1).numpy()
    lambd6 = 0.8
    input_dict6 = {"lambd": lambd6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(10).numpy()
    lambd7 = 0.01
    input_dict7 = {"lambd": lambd7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.tensor([2.5, -3.1, 1.7, -0.9]).numpy()
    lambd8 = 1.5
    input_dict8 = {"lambd": lambd8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(2, 2).numpy()
    lambd9 = -0.3
    input_dict9 = {"lambd": lambd9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.tensor([0.0, 0.0, 0.0]).numpy()
    lambd10 = 0.6
    input_dict10 = {"lambd": lambd10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.Hardshrink"] = hardshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Hardshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Hardshrink'.")


check_valid('torch.nn.Hardshrink', generated_inputs['torch.nn.Hardshrink'], lib="torch", suffix=0)
