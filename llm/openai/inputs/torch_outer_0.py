
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def outer_inputs():
    list_of_inputs = []

    input1 = torch.arange(1., 5.).numpy()
    vec2_1 = torch.arange(1., 4.).numpy()
    out1 = np.zeros((4, 3))

    input_dict1 = {
        "input": input1,
        "vec2": vec2_1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([-1.0, 2.0, -3.0]).numpy()
    vec2_2 = torch.tensor([0.5, -1.0, 2.0]).numpy()
    out2 = np.zeros((3, 3))

    input_dict2 = {
        "input": input2,
        "vec2": vec2_2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2).numpy()
    vec2_3 = torch.randn(3).numpy()
    out3 = np.zeros((2, 3))

    input_dict3 = {
        "input": input3,
        "vec2": vec2_3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([1.0]).numpy()
    vec2_4 = torch.tensor([2.0, 3.0, 4.0]).numpy()
    out4 = np.zeros((1, 3))

    input_dict4 = {
        "input": input4,
        "vec2": vec2_4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([5.0, 6.0]).numpy()
    vec2_5 = torch.tensor([7.0]).numpy()
    out5 = np.zeros((2, 1))

    input_dict5 = {
        "input": input5,
        "vec2": vec2_5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.zeros(4).numpy()
    vec2_6 = torch.ones(2).numpy()
    out6 = np.zeros((4, 2))

    input_dict6 = {
        "input": input6,
        "vec2": vec2_6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.arange(2.0, 6.0).numpy()
    vec2_7 = torch.arange(3.0, 7.0).numpy()
    out7 = np.zeros((4, 4))

    input_dict7 = {
        "input": input7,
        "vec2": vec2_7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.tensor([1.5, 2.5, 3.5]).numpy()
    vec2_8 = torch.tensor([4.0, 5.0]).numpy()
    out8 = np.zeros((3, 2))

    input_dict8 = {
        "input": input8,
        "vec2": vec2_8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([-0.1, 0.2, -0.3]).numpy()
    vec2_9 = torch.tensor([0.5, -0.6]).numpy()
    out9 = np.zeros((3, 2))

    input_dict9 = {
        "input": input9,
        "vec2": vec2_9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.tensor([10.0, -5.0, 2.0]).numpy()
    vec2_10 = torch.tensor([1.0, -2.0, 3.0]).numpy()
    out10 = np.zeros((3, 3))

    input_dict10 = {
        "input": input10,
        "vec2": vec2_10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.outer"] = outer_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.outer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.outer'.")


check_valid('torch.outer', generated_inputs['torch.outer'], lib="torch", suffix=0)
