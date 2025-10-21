
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def baddbmm_inputs():
    list_of_inputs = []
    input1 = np.random.rand(10, 3, 5).astype(np.float32)
    batch1_1 = np.random.rand(10, 3, 4).astype(np.float32)
    batch2_1 = np.random.rand(10, 4, 5).astype(np.float32)
    beta1 = 1.0
    alpha1 = 1.0
    out1 = np.zeros((10, 3, 5)).astype(np.float32)

    input_dict1 = {
        "input": input1,
        "batch1": batch1_1,
        "batch2": batch2_1,
        "beta": beta1,
        "alpha": alpha1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(5, 2, 3).astype(np.float32)
    batch1_2 = np.random.rand(5, 2, 4).astype(np.float32)
    batch2_2 = np.random.rand(5, 4, 3).astype(np.float32)
    beta2 = 0.5
    alpha2 = 2.0
    out2 = np.zeros((5, 2, 3)).astype(np.float32)

    input_dict2 = {
        "input": input2,
        "batch1": batch1_2,
        "batch2": batch2_2,
        "beta": beta2,
        "alpha": alpha2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(8, 4, 2).astype(np.float32)
    batch1_3 = np.random.rand(8, 4, 3).astype(np.float32)
    batch2_3 = np.random.rand(8, 3, 2).astype(np.float32)
    beta3 = -1.0
    alpha3 = -0.5
    out3 = np.zeros((8, 4, 2)).astype(np.float32)

    input_dict3 = {
        "input": input3,
        "batch1": batch1_3,
        "batch2": batch2_3,
        "beta": beta3,
        "alpha": alpha3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 2, 2).astype(np.float32)
    batch1_4 = np.random.rand(2, 2, 2).astype(np.float32)
    batch2_4 = np.random.rand(2, 2, 2).astype(np.float32)
    beta4 = 1.0
    alpha4 = 1.0
    out4 = np.zeros((2, 2, 2)).astype(np.float32)

    input_dict4 = {
        "input": input4,
        "batch1": batch1_4,
        "batch2": batch2_4,
        "beta": beta4,
        "alpha": alpha4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(3, 1, 4).astype(np.float32)
    batch1_5 = np.random.rand(3, 1, 2).astype(np.float32)
    batch2_5 = np.random.rand(3, 2, 4).astype(np.float32)
    beta5 = 0.0
    alpha5 = 1.5
    out5 = np.zeros((3, 1, 4)).astype(np.float32)

    input_dict5 = {
        "input": input5,
        "batch1": batch1_5,
        "batch2": batch2_5,
        "beta": beta5,
        "alpha": alpha5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(4, 2, 1).astype(np.float32)
    batch1_6 = np.random.rand(4, 2, 3).astype(np.float32)
    batch2_6 = np.random.rand(4, 3, 1).astype(np.float32)
    beta6 = 2.0
    alpha6 = 0.75
    out6 = np.zeros((4, 2, 1)).astype(np.float32)

    input_dict6 = {
        "input": input6,
        "batch1": batch1_6,
        "batch2": batch2_6,
        "beta": beta6,
        "alpha": alpha6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.rand(6, 3, 2).astype(np.float32)
    batch1_7 = np.random.rand(6, 3, 5).astype(np.float32)
    batch2_7 = np.random.rand(6, 5, 2).astype(np.float32)
    beta7 = -0.2
    alpha7 = 1.1
    out7 = np.zeros((6, 3, 2)).astype(np.float32)

    input_dict7 = {
        "input": input7,
        "batch1": batch1_7,
        "batch2": batch2_7,
        "beta": beta7,
        "alpha": alpha7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.random.rand(7, 4, 3).astype(np.float32)
    batch1_8 = np.random.rand(7, 4, 1).astype(np.float32)
    batch2_8 = np.random.rand(7, 1, 3).astype(np.float32)
    beta8 = 1.0
    alpha8 = 0.0
    out8 = np.zeros((7, 4, 3)).astype(np.float32)

    input_dict8 = {
        "input": input8,
        "batch1": batch1_8,
        "batch2": batch2_8,
        "beta": beta8,
        "alpha": alpha8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(9, 2, 5).astype(np.float32)
    batch1_9 = np.random.rand(9, 2, 3).astype(np.float32)
    batch2_9 = np.random.rand(9, 3, 5).astype(np.float32)
    beta9 = 0.8
    alpha9 = -1.2
    out9 = np.zeros((9, 2, 5)).astype(np.float32)

    input_dict9 = {
        "input": input9,
        "batch1": batch1_9,
        "batch2": batch2_9,
        "beta": beta9,
        "alpha": alpha9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(1, 5, 2).astype(np.float32)
    batch1_10 = np.random.rand(1, 5, 1).astype(np.float32)
    batch2_10 = np.random.rand(1, 1, 2).astype(np.float32)
    beta10 = -0.5
    alpha10 = 0.9
    out10 = np.zeros((1, 5, 2)).astype(np.float32)

    input_dict10 = {
        "input": input10,
        "batch1": batch1_10,
        "batch2": batch2_10,
        "beta": beta10,
        "alpha": alpha10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.baddbmm"] = baddbmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.baddbmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.baddbmm'.")


check_valid('torch.baddbmm', generated_inputs['torch.baddbmm'], lib="torch", suffix=0)
