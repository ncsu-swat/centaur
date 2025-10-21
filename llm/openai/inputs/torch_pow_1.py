
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_pow_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(3).numpy()
    exponent1 = 2.0
    out1 = torch.zeros(3).numpy()
    
    input_dict1 = {
        "input": input1,
        "exponent": exponent1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    exponent2 = 0.5
    out2 = torch.zeros(4).numpy()

    input_dict2 = {
        "input": input2,
        "exponent": exponent2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 2).numpy()
    exponent3 = 3.0
    out3 = torch.zeros((2, 2)).numpy()

    input_dict3 = {
        "input": input3,
        "exponent": exponent3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    exponent4 = 2.0
    out4 = torch.zeros(3).numpy()

    input_dict4 = {
        "input": input4,
        "exponent": exponent4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([0.0, 1.0, 2.0]).numpy()
    exponent5 = 4.0
    out5 = torch.zeros(3).numpy()

    input_dict5 = {
        "input": input5,
        "exponent": exponent5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 3, 4).numpy()
    exponent6 = 1.5
    out6 = torch.zeros((2, 3, 4)).numpy()

    input_dict6 = {
        "input": input6,
        "exponent": exponent6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([1.0, -1.0, 2.0]).numpy()
    exponent7 = 0.0
    out7 = torch.zeros(3).numpy()
    
    input_dict7 = {
        "input": input7,
        "exponent": exponent7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(5).numpy()
    exponent8 = -1.0
    out8 = torch.zeros(5).numpy()

    input_dict8 = {
        "input": input8,
        "exponent": exponent8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([2.0, 4.0, 6.0]).numpy()
    exponent9 = 1.0
    out9 = torch.zeros(3).numpy()

    input_dict9 = {
        "input": input9,
        "exponent": exponent9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.randn(1, 1).numpy()
    exponent10 = 5.0
    out10 = torch.zeros((1, 1)).numpy()

    input_dict10 = {
        "input": input10,
        "exponent": exponent10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.pow_1"] = torch_pow_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.pow_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pow_1'.")


check_valid('torch.pow', generated_inputs['torch.pow_1'], lib="torch", suffix=1)
