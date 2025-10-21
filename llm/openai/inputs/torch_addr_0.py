
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def addr_inputs():
    list_of_inputs = []
    input1 = torch.zeros(3, 2).numpy()
    vec1_1 = torch.arange(1., 4.).numpy()
    vec2_1 = torch.arange(1., 3.).numpy()
    beta1 = 1.0
    alpha1 = 1.0
    out1 = torch.empty(3, 2).numpy()
    
    input_dict1 = {
        "input": input1,
        "vec1": vec1_1,
        "vec2": vec2_1,
        "beta": beta1,
        "alpha": alpha1,
        "out": out1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 4).numpy()
    vec1_2 = torch.randn(2).numpy()
    vec2_2 = torch.randn(4).numpy()
    beta2 = 0.5
    alpha2 = 2.0
    out2 = torch.empty(2, 4).numpy()

    input_dict2 = {
        "input": input2,
        "vec1": vec1_2,
        "vec2": vec2_2,
        "beta": beta2,
        "alpha": alpha2,
        "out": out2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.ones(1, 5).numpy()
    vec1_3 = torch.tensor([2.0]).numpy()
    vec2_3 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    beta3 = 0.0
    alpha3 = 1.5
    out3 = torch.empty(1, 5).numpy()

    input_dict3 = {
        "input": input3,
        "vec1": vec1_3,
        "vec2": vec2_3,
        "beta": beta3,
        "alpha": alpha3,
        "out": out3
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(4, 4).numpy()
    vec1_4 = torch.randn(4).numpy()
    vec2_4 = torch.randn(4).numpy()
    beta4 = -1.0
    alpha4 = -0.5
    out4 = torch.empty(4, 4).numpy()
    
    input_dict4 = {
        "input": input4,
        "vec1": vec1_4,
        "vec2": vec2_4,
        "beta": beta4,
        "alpha": alpha4,
        "out": out4
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.addr"] = addr_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.addr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addr'.")


check_valid('torch.addr', generated_inputs['torch.addr'], lib="torch", suffix=0)
