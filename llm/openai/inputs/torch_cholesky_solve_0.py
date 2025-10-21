
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cholesky_solve_inputs():
    list_of_inputs = []
    
    input1 = np.array([[4, 12, -16],
                       [12, 37, -43],
                       [-16, -43, 98]], dtype=np.float64)
    L1 = np.array([[2, 0, 0],
                   [1, 3, 0],
                   [-2, 4, 5]], dtype=np.float64)
    upper1 = False
    out1 = np.empty((3, 3), dtype=np.float64)

    input_dict1 = {
        "input": input1,
        "L": L1,
        "upper": upper1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 2, 3],
                       [2, 5, 7],
                       [3, 7, 13]], dtype=np.float32)
    L2 = np.array([[1, 0, 0],
                   [1, 2, 0],
                   [1, 3, 3]], dtype=np.float32)
    upper2 = False
    out2 = np.empty((3, 3), dtype=np.float32)

    input_dict2 = {
        "input": input2,
        "L": L2,
        "upper": upper2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[25, 15, -5],
                       [15, 18, 0],
                       [-5, 0, 11]], dtype=np.float64)
    L3 = np.array([[5, 0, 0],
                   [3, 2, 0],
                   [-1, 0, 3]], dtype=np.float64)
    upper3 = True
    out3 = np.empty((3, 3), dtype=np.float64)

    input_dict3 = {
        "input": input3,
        "L": L3,
        "upper": upper3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[1.0, 0.0],
                       [0.0, 1.0]], dtype=np.float64)
    L4 = np.array([[1.0, 0.0],
                   [0.0, 1.0]], dtype=np.float64)
    upper4 = False
    out4 = np.empty((2, 2), dtype=np.float64)
    
    input_dict4 = {
        "input": input4,
        "L": L4,
        "upper": upper4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[9, 3, 0],
                       [3, 3, 1],
                       [0, 1, 4]], dtype=np.float32)
    L5 = np.array([[3, 0, 0],
                   [1, 1, 0],
                   [0, 1, 2]], dtype=np.float32)
    upper5 = True
    out5 = np.empty((3, 3), dtype=np.float32)

    input_dict5 = {
        "input": input5,
        "L": L5,
        "upper": upper5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    L6 = np.array([[1, 0], [2, 1]], dtype=np.float64)
    upper6 = False
    out6 = np.empty((2, 2), dtype=np.float64)

    input_dict6 = {
        "input": input6,
        "L": L6,
        "upper": upper6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[5, 2, 1], [2, 8, 3], [1, 3, 6]], dtype=np.float32)
    L7 = np.array([[np.sqrt(5), 0, 0], [np.sqrt(2), np.sqrt(6), 0], [np.sqrt(1), np.sqrt(1), np.sqrt(3)]], dtype=np.float32)
    upper7 = False
    out7 = np.empty((3, 3), dtype=np.float32)

    input_dict7 = {
        "input": input7,
        "L": L7,
        "upper": upper7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[10, 5], [5, 10]], dtype=np.float64)
    L8 = np.array([[np.sqrt(10), 0], [np.sqrt(5), np.sqrt(5)]], dtype=np.float64)
    upper8 = False
    out8 = np.empty((2, 2), dtype=np.float64)

    input_dict8 = {
        "input": input8,
        "L": L8,
        "upper": upper8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[16, 4, 4], [4, 16, 4], [4, 4, 16]], dtype=np.float32)
    L9 = np.array([[4, 0, 0], [1, 4, 0], [1, 1, 4]], dtype=np.float32)
    upper9 = True
    out9 = np.empty((3, 3), dtype=np.float32)

    input_dict9 = {
        "input": input9,
        "L": L9,
        "upper": upper9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[7, 2, 1], [2, 6, 2], [1, 2, 5]], dtype=np.float64)
    L10 = np.array([[np.sqrt(7), 0, 0], [np.sqrt(2), np.sqrt(4), 0], [np.sqrt(1), np.sqrt(1), np.sqrt(3)]], dtype=np.float64)
    upper10 = False
    out10 = np.empty((3, 3), dtype=np.float64)

    input_dict10 = {
        "input": input10,
        "L": L10,
        "upper": upper10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.cholesky_solve"] = cholesky_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.cholesky_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cholesky_solve'.")


check_valid('torch.cholesky_solve', generated_inputs['torch.cholesky_solve'], lib="torch", suffix=0)
