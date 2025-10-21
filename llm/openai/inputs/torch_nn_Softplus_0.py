
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def softplus_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {"beta": 1.0, "threshold": 20.0, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {"beta": 0.5, "threshold": 10.0, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    input_dict3 = {"beta": 2.0, "threshold": 30.0, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1).numpy()
    input_dict4 = {"beta": 1.5, "threshold": 15.0, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(5).numpy()
    input_dict5 = {"beta": 0.2, "threshold": 5.0, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 2, 2).numpy()
    input_dict6 = {"beta": 3.0, "threshold": 40.0, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([100.0, 200.0, 300.0]).numpy()
    input_dict7 = {"beta": 1.0, "threshold": 20.0, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(4, 4).numpy()
    input_dict8 = {"beta": 0.8, "threshold": 8.0, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([-5.0, 0.0, 5.0]).numpy()
    input_dict9 = {"beta": 1.2, "threshold": 25.0, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(10, 10, 10).numpy()
    input_dict10 = {"beta": 2.5, "threshold": 35.0, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.Softplus"] = softplus_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softplus'.")


check_valid('torch.nn.Softplus', generated_inputs['torch.nn.Softplus'], lib="torch", suffix=0)
