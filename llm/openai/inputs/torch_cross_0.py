
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_cross_inputs():
    list_of_inputs = []

    input1 = np.random.rand(3)
    other1 = np.random.rand(3)
    dim1 = None
    out1 = np.zeros(3)

    input_dict1 = {
        "input": input1,
        "other": other1,
        "dim": dim1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(4, 3)
    other2 = np.random.rand(4, 3)
    dim2 = 1
    out2 = np.zeros((4, 3))

    input_dict2 = {
        "input": input2,
        "other": other2,
        "dim": dim2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(2, 2, 3)
    other3 = np.random.rand(2, 2, 3)
    dim3 = 2
    out3 = np.zeros((2, 2, 3))

    input_dict3 = {
        "input": input3,
        "other": other3,
        "dim": dim3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(3)
    other4 = np.random.rand(3)
    dim4 = None
    out4 = np.zeros(3)

    input_dict4 = {
        "input": input4,
        "other": other4,
        "dim": dim4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(4, 3)
    other5 = np.random.rand(4, 3)
    dim5 = None
    out5 = np.zeros((4, 3))

    input_dict5 = {
        "input": input5,
        "other": other5,
        "dim": dim5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(3)
    other6 = np.random.rand(3)
    dim6 = None
    out6 = np.zeros(3)

    input_dict6 = {
        "input": input6,
        "other": other6,
        "dim": dim6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(2, 3)
    other7 = np.random.rand(2, 3)
    dim7 = 1
    out7 = np.zeros((2, 3))

    input_dict7 = {
        "input": input7,
        "other": other7,
        "dim": dim7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    

    return list_of_inputs

generated_inputs["torch.cross"] = torch_cross_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cross'.")


check_valid('torch.cross', generated_inputs['torch.cross'], lib="torch", suffix=0)
