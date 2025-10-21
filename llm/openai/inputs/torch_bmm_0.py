
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bmm_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(10, 3, 4).astype(np.float32)
    mat2_1 = np.random.rand(10, 4, 5).astype(np.float32)
    out1 = np.zeros((10, 3, 5)).astype(np.float32)
    
    input_dict1 = {
        "input": input1,
        "mat2": mat2_1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(5, 2, 3).astype(np.float32)
    mat2_2 = np.random.rand(5, 3, 6).astype(np.float32)
    out2 = np.zeros((5, 2, 6)).astype(np.float32)
    
    input_dict2 = {
        "input": input2,
        "mat2": mat2_2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(2, 4, 2).astype(np.float32)
    mat2_3 = np.random.rand(2, 2, 5).astype(np.float32)
    out3 = np.zeros((2, 4, 5)).astype(np.float32)
    
    input_dict3 = {
        "input": input3,
        "mat2": mat2_3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1, 5, 5).astype(np.float32)
    mat2_4 = np.random.rand(1, 5, 2).astype(np.float32)
    out4 = np.zeros((1, 5, 2)).astype(np.float32)

    input_dict4 = {
        "input": input4,
        "mat2": mat2_4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(3, 2, 2).astype(np.float32)
    mat2_5 = np.random.rand(3, 2, 3).astype(np.float32)
    out5 = np.zeros((3, 2, 3)).astype(np.float32)

    input_dict5 = {
        "input": input5,
        "mat2": mat2_5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(4, 1, 3).astype(np.float32)
    mat2_6 = np.random.rand(4, 3, 1).astype(np.float32)
    out6 = np.zeros((4, 1, 1)).astype(np.float32)
    
    input_dict6 = {
        "input": input6,
        "mat2": mat2_6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.rand(2, 2, 1).astype(np.float32)
    mat2_7 = np.random.rand(2, 1, 2).astype(np.float32)
    out7 = np.zeros((2, 2, 2)).astype(np.float32)

    input_dict7 = {
        "input": input7,
        "mat2": mat2_7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(1, 3, 3).astype(np.float32)
    mat2_8 = np.random.rand(1, 3, 3).astype(np.float32)
    out8 = np.zeros((1, 3, 3)).astype(np.float32)
    
    input_dict8 = {
        "input": input8,
        "mat2": mat2_8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(6, 2, 4).astype(np.float32)
    mat2_9 = np.random.rand(6, 4, 1).astype(np.float32)
    out9 = np.zeros((6, 2, 1)).astype(np.float32)
    
    input_dict9 = {
        "input": input9,
        "mat2": mat2_9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 5, 1).astype(np.float32)
    mat2_10 = np.random.rand(2, 1, 5).astype(np.float32)
    out10 = np.zeros((2, 5, 5)).astype(np.float32)
    
    input_dict10 = {
        "input": input10,
        "mat2": mat2_10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.bmm"] = bmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bmm'.")


check_valid('torch.bmm', generated_inputs['torch.bmm'], lib="torch", suffix=0)
