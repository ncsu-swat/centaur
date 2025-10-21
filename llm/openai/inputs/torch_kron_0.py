
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def kron_inputs():
    list_of_inputs = []
    
    input1 = torch.eye(2).numpy()
    other1 = torch.ones(2, 2).numpy()
    out1 = np.zeros((4, 4))
    
    input_dict1 = {
        "input": input1,
        "other": other1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.eye(2).numpy()
    other2 = torch.arange(1, 5).reshape(2, 2).numpy()
    out2 = np.zeros((4, 4))
    
    input_dict2 = {
        "input": input2,
        "other": other2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(3, 3).astype(np.float32)
    other3 = np.random.rand(2, 2).astype(np.float32)
    out3 = np.zeros((6, 6))
    
    input_dict3 = {
        "input": input3,
        "other": other3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.ones((2, 3)).astype(np.float32)
    other4 = np.zeros((4, 1)).astype(np.float32)
    out4 = np.zeros((8, 3))

    input_dict4 = {
        "input": input4,
        "other": other4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[-1, 2], [3, -4]]).astype(np.float32)
    other5 = np.array([[5, -6], [-7, 8]]).astype(np.float32)
    out5 = np.zeros((4, 4))

    input_dict5 = {
        "input": input5,
        "other": other5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(1, 5).astype(np.float32)
    other6 = np.random.rand(2, 2).astype(np.float32)
    out6 = np.zeros((2, 10))

    input_dict6 = {
        "input": input6,
        "other": other6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(5, 1).astype(np.float32)
    other7 = np.random.rand(2, 2).astype(np.float32)
    out7 = np.zeros((10, 2))

    input_dict7 = {
        "input": input7,
        "other": other7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(2, 2, 2).astype(np.float32)
    other8 = np.random.rand(2, 2, 2).astype(np.float32)
    out8 = np.zeros((8, 8, 8))

    input_dict8 = {
        "input": input8,
        "other": other8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(1, 1, 1).astype(np.float32)
    other9 = np.random.rand(2, 2, 2).astype(np.float32)
    out9 = np.zeros((2, 2, 2))

    input_dict9 = {
        "input": input9,
        "other": other9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.random.rand(2, 2).astype(np.complex128)
    other10 = np.random.rand(2, 2).astype(np.complex128)
    out10 = np.zeros((4, 4), dtype=np.complex128)

    input_dict10 = {
        "input": input10,
        "other": other10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.kron"] = kron_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.kron' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.kron'.")


check_valid('torch.kron', generated_inputs['torch.kron'], lib="torch", suffix=0)
