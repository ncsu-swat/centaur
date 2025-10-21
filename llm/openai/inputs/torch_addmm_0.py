
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def addmm_inputs():
    list_of_inputs = []
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    mat1_1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    mat2_1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    beta1 = 1.0
    alpha1 = 1.0
    out1 = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict1 = {
        "input": input1,
        "mat1": mat1_1,
        "mat2": mat2_1,
        "beta": beta1,
        "alpha": alpha1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0, 3.0]])
    mat1_2 = np.array([[4.0], [5.0], [6.0]])
    mat2_2 = np.array([[7.0, 8.0, 9.0]])
    beta2 = 0.5
    alpha2 = 2.0
    out2 = np.array([[0.0, 0.0, 0.0]])

    input_dict2 = {
        "input": input2,
        "mat1": mat1_2,
        "mat2": mat2_2,
        "beta": beta2,
        "alpha": alpha2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    mat1_3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    mat2_3 = np.array([[5.0, 6.0], [7.0, 8.0]])
    beta3 = 0.0
    alpha3 = 1.0
    out3 = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict3 = {
        "input": input3,
        "mat1": mat1_3,
        "mat2": mat2_3,
        "beta": beta3,
        "alpha": alpha3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 2)
    mat1_4 = np.random.rand(2, 2)
    mat2_4 = np.random.rand(2, 2)
    beta4 = 1.5
    alpha4 = 0.5
    out4 = np.zeros((2, 2))

    input_dict4 = {
        "input": input4,
        "mat1": mat1_4,
        "mat2": mat2_4,
        "beta": beta4,
        "alpha": alpha4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[1.0, 2.0, 3.0, 4.0]])
    mat1_5 = np.array([[5.0], [6.0], [7.0], [8.0]])
    mat2_5 = np.array([[9.0, 10.0, 11.0, 12.0]])
    beta5 = 1.0
    alpha5 = 1.0
    out5 = np.array([[0.0, 0.0, 0.0, 0.0]])

    input_dict5 = {
        "input": input5,
        "mat1": mat1_5,
        "mat2": mat2_5,
        "beta": beta5,
        "alpha": alpha5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[1.0, 2.0], [3.0, 4.0]])
    mat1_6 = np.array([[1.0, 2.0], [3.0, 4.0]])
    mat2_6 = np.array([[5.0, 6.0], [7.0, 8.0]])
    beta6 = 1.0
    alpha6 = 1.0
    out6 = np.array([[1.0, 2.0], [3.0, 4.0]])

    input_dict6 = {
        "input": input6,
        "mat1": mat1_6,
        "mat2": mat2_6,
        "beta": beta6,
        "alpha": alpha6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(3, 3)
    mat1_7 = np.random.rand(3, 3)
    mat2_7 = np.random.rand(3, 3)
    beta7 = 0.7
    alpha7 = 1.3
    out7 = np.zeros((3, 3))

    input_dict7 = {
        "input": input7,
        "mat1": mat1_7,
        "mat2": mat2_7,
        "beta": beta7,
        "alpha": alpha7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1.0]])
    mat1_8 = np.array([[2.0]])
    mat2_8 = np.array([[3.0]])
    beta8 = 1.0
    alpha8 = 1.0
    out8 = np.array([[0.0]])

    input_dict8 = {
        "input": input8,
        "mat1": mat1_8,
        "mat2": mat2_8,
        "beta": beta8,
        "alpha": alpha8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1.0, 2.0], [3.0, 4.0]])
    mat1_9 = np.array([[5.0, 6.0], [7.0, 8.0]])
    mat2_9 = np.array([[9.0, 10.0], [11.0, 12.0]])
    beta9 = -1.0
    alpha9 = -1.0
    out9 = np.array([[0.0, 0.0], [0.0, 0.0]])

    input_dict9 = {
        "input": input9,
        "mat1": mat1_9,
        "mat2": mat2_9,
        "beta": beta9,
        "alpha": alpha9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(4, 4)
    mat1_10 = np.random.rand(4, 4)
    mat2_10 = np.random.rand(4, 4)
    beta10 = 2.0
    alpha10 = 0.5
    out10 = np.zeros((4, 4))

    input_dict10 = {
        "input": input10,
        "mat1": mat1_10,
        "mat2": mat2_10,
        "beta": beta10,
        "alpha": alpha10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.addmm"] = addmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.addmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmm'.")


check_valid('torch.addmm', generated_inputs['torch.addmm'], lib="torch", suffix=0)
