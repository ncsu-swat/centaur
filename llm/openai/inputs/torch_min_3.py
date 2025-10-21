
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_min_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0])
    other1 = np.array([0.5, 1.5, 2.5])
    out1 = np.array([])

    input_dict1 = {
        "input": input1,
        "other": other1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[-1.0, 0.5], [2.0, -3.0]])
    other2 = np.array([[0.0, -1.0], [-2.0, 1.0]])
    out2 = np.array([])

    input_dict2 = {
        "input": input2,
        "other": other2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    other3 = np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])
    out3 = np.array([])

    input_dict3 = {
        "input": input3,
        "other": other3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([10, -5, 2, 8])
    other4 = np.array([5, 0, -1, 3])
    out4 = np.array([])

    input_dict4 = {
        "input": input4,
        "other": other4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    other5 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    out5 = np.array([])
    
    input_dict5 = {
        "input": input5,
        "other": other5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-1, -2, -3], [-4, -5, -6]])
    other6 = np.array([[-7, -8, -9], [-10, -11, -12]])
    out6 = np.array([])

    input_dict6 = {
        "input": input6,
        "other": other6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.5, 2.5, 3.5])
    other7 = np.array([0.5, 1.5, 2.5])
    out7 = np.array([])

    input_dict7 = {
        "input": input7,
        "other": other7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1, 2], [3, 4]])
    other8 = np.array([[5, 6], [7, 8]])
    out8 = np.array([])
    
    input_dict8 = {
        "input": input8,
        "other": other8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[-2.0, 1.0], [0.5, -3.0]])
    other9 = np.array([[1.0, -2.0], [-3.0, 0.5]])
    out9 = np.array([])

    input_dict9 = {
        "input": input9,
        "other": other9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([100, -200, 300])
    other10 = np.array([50, -100, 150])
    out10 = np.array([])

    input_dict10 = {
        "input": input10,
        "other": other10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.min_3"] = torch_min_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.min_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.min_3'.")


check_valid('torch.min', generated_inputs['torch.min_3'], lib="torch", suffix=3)
