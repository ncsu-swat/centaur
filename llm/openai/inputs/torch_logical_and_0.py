
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logical_and_inputs():
    list_of_inputs = []
    
    input1 = np.array([True, False, True])
    input2 = np.array([True, True, False])
    out1 = np.array([])
    
    input_dict = {
        "input": input1,
        "other": input2,
        "out": out1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input3 = np.array([[True, False], [False, True]])
    input4 = np.array([[False, True], [True, False]])
    out2 = np.array([])
    
    input_dict = {
        "input": input3,
        "other": input4,
        "out": out2
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input5 = np.array([1, 0, 1])
    input6 = np.array([1, 1, 0])
    out3 = np.array([])
    
    input_dict = {
        "input": input5.astype(bool),
        "other": input6.astype(bool),
        "out": out3
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input7 = np.array([[1, 0], [0, 1]])
    input8 = np.array([[0, 1], [1, 0]])
    out4 = np.array([])
    
    input_dict = {
        "input": input7.astype(bool),
        "other": input8.astype(bool),
        "out": out4
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input9 = np.array([True, True, True])
    input10 = np.array([False, False, False])
    out5 = np.array([])
    
    input_dict = {
        "input": input9,
        "other": input10,
        "out": out5
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input11 = np.array([False, False, False])
    input12 = np.array([True, True, True])
    out6 = np.array([])

    input_dict = {
        "input": input11,
        "other": input12,
        "out": out6
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input13 = np.array([True])
    input14 = np.array([False])
    out7 = np.array([])

    input_dict = {
        "input": input13,
        "other": input14,
        "out": out7
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input15 = np.array([False])
    input16 = np.array([True])
    out8 = np.array([])

    input_dict = {
        "input": input15,
        "other": input16,
        "out": out8
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input17 = np.array([True, False, True, False])
    input18 = np.array([False, True, False, True])
    out9 = np.array([])

    input_dict = {
        "input": input17,
        "other": input18,
        "out": out9
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input19 = np.array([[True, False, True], [False, True, False]])
    input20 = np.array([[False, True, False], [True, False, True]])
    out10 = np.array([])
    
    input_dict = {
        "input": input19,
        "other": input20,
        "out": out10
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.logical_and"] = logical_and_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logical_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logical_and'.")


check_valid('torch.logical_and', generated_inputs['torch.logical_and'], lib="torch", suffix=0)
