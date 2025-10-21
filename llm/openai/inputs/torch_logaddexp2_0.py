
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logaddexp2_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    other1 = np.array([0.1, 0.2, 0.3])
    out1 = np.array([])
    
    input_dict1 = {
        "input": input1,
        "other": other1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other2 = np.array([[0.1, 0.2], [0.3, 0.4]])
    out2 = np.array([])
    
    input_dict2 = {
        "input": input2,
        "other": other2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([-1.0, -2.0, -3.0])
    other3 = np.array([-0.1, -0.2, -0.3])
    out3 = np.array([])
    
    input_dict3 = {
        "input": input3,
        "other": other3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1.0])
    other4 = np.array([0.1])
    out4 = np.array([])
    
    input_dict4 = {
        "input": input4,
        "other": other4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1.0, 2.0, 3.0])
    other5 = np.array([4.0, 5.0, 6.0])
    out5 = np.array([0.0, 0.0, 0.0])

    input_dict5 = {
        "input": input5,
        "other": other5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    other6 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    out6 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    input_dict6 = {
        "input": input6,
        "other": other6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1e9, 1e9])
    other7 = np.array([1e9, 1e9])
    out7 = np.array([])
    
    input_dict7 = {
        "input": input7,
        "other": other7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([-1e9, -1e9])
    other8 = np.array([-1e9, -1e9])
    out8 = np.array([])
    
    input_dict8 = {
        "input": input8,
        "other": other8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([0.0, 0.0, 0.0])
    other9 = np.array([0.0, 0.0, 0.0])
    out9 = np.array([])
    
    input_dict9 = {
        "input": input9,
        "other": other9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([1.5, 2.5, 3.5])
    other10 = np.array([0.5, 1.5, 2.5])
    out10 = np.array([])
    
    input_dict10 = {
        "input": input10,
        "other": other10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.logaddexp2"] = logaddexp2_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logaddexp2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logaddexp2'.")


check_valid('torch.logaddexp2', generated_inputs['torch.logaddexp2'], lib="torch", suffix=0)
