
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cholesky_inputs():
    list_of_inputs = []
    
    input1 = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.float64)
    upper1 = False
    out1 = np.zeros((3, 3), dtype=np.float64)
    
    input_dict1 = {
        "input": input1,
        "upper": upper1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]], dtype=np.float64)
    upper2 = True
    out2 = np.zeros((3, 3), dtype=np.float64)
    
    input_dict2 = {
        "input": input2,
        "upper": upper2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1, 2, 3], [2, 5, 7], [3, 7, 11]], dtype=np.float64)
    upper3 = False
    out3 = np.zeros((3, 3), dtype=np.float64)
    
    input_dict3 = {
        "input": input3,
        "upper": upper3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[10, -1, 2], [-1, 5, -1], [2, -1, 8]], dtype=np.float64)
    upper4 = True
    out4 = np.zeros((3, 3), dtype=np.float64)

    input_dict4 = {
        "input": input4,
        "upper": upper4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[9, 3, 0], [3, 7, 1], [0, 1, 5]], dtype=np.float64)
    upper5 = False
    out5 = np.zeros((3, 3), dtype=np.float64)

    input_dict5 = {
        "input": input5,
        "upper": upper5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    upper6 = True
    out6 = np.zeros((3, 3), dtype=np.float64)

    input_dict6 = {
        "input": input6,
        "upper": upper6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(4, 4)
    input7 = input7 @ input7.T + 0.1
    upper7 = False
    out7 = np.zeros((4, 4), dtype=np.float64)

    input_dict7 = {
        "input": input7,
        "upper": upper7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(2, 2)
    input8 = input8 @ input8.T + 0.1
    upper8 = True
    out8 = np.zeros((2, 2), dtype=np.float64)

    input_dict8 = {
        "input": input8,
        "upper": upper8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[2, 1], [1, 2]], dtype=np.float64)
    upper9 = False
    out9 = np.zeros((2, 2), dtype=np.float64)

    input_dict9 = {
        "input": input9,
        "upper": upper9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[5, 2, 1], [2, 8, 3], [1, 3, 6]], dtype=np.float64)
    upper10 = True
    out10 = np.zeros((3, 3), dtype=np.float64)

    input_dict10 = {
        "input": input10,
        "upper": upper10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.cholesky"] = cholesky_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cholesky' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cholesky'.")


check_valid('torch.cholesky', generated_inputs['torch.cholesky'], lib="torch", suffix=0)
