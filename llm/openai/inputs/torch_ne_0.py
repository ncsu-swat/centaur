
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_ne_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 3])
    tensor1 = np.array([1, 2, 3])
    tensor2 = np.array([4, 5, 6])
    out1 = np.array([])

    input_dict1 = {
        "input": input1,
        "other": tensor1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 2], [3, 4]])
    tensor1 = np.array([[1, 1], [4, 4]])
    tensor2 = np.array([[0, 0], [0, 0]])
    out2 = np.array([])

    input_dict2 = {
        "input": input2,
        "other": tensor1,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1, 2], [3, -4]])
    tensor1 = np.array([[1, -2], [-3, 4]])
    tensor2 = np.array([[0, 0], [0, 0]])
    out3 = np.array([])

    input_dict3 = {
        "input": input3,
        "other": tensor1,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0, 2.0, 3.0])
    tensor1 = np.array([1.0, 2.0, 3.0])
    tensor2 = np.array([4.0, 5.0, 6.0])
    out4 = np.array([])

    input_dict4 = {
        "input": input4,
        "other": tensor1,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tensor1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tensor2 = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    out5 = np.array([])

    input_dict5 = {
        "input": input5,
        "other": tensor1,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1, 2, 3, 4, 5])
    tensor1 = np.array([5, 4, 3, 2, 1])
    tensor2 = np.array([0, 0, 0, 0, 0])
    out6 = np.array([])

    input_dict6 = {
        "input": input6,
        "other": tensor1,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[1, 2, 3], [4, 5, 6]])
    tensor1 = np.array([[7, 8, 9], [10, 11, 12]])
    tensor2 = np.array([[1, 2, 3], [4, 5, 6]])
    out7 = np.array([])

    input_dict7 = {
        "input": input7,
        "other": tensor1,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1, 1, 1])
    tensor1 = np.array([1, 1, 1])
    tensor2 = np.array([2, 2, 2])
    out8 = np.array([])

    input_dict8 = {
        "input": input8,
        "other": tensor1,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([[1,2],[3,4]])
    tensor1 = np.array([[5,6],[7,8]])
    tensor2 = np.array([[1,2],[3,4]])
    out9 = np.array([])
    
    input_dict9 = {
        "input": input9,
        "other": tensor1,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0, 0, 0])
    tensor1 = np.array([1, 2, 3])
    tensor2 = np.array([0, 0, 0])
    out10 = np.array([])

    input_dict10 = {
        "input": input10,
        "other": tensor1,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.ne"] = torch_ne_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ne' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ne'.")


check_valid('torch.ne', generated_inputs['torch.ne'], lib="torch", suffix=0)
