
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def xlogy_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out1 = np.array([], dtype=np.float32)
    
    input_dict1 = {
        "x": input1,
        "y": input2,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input4 = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    out2 = np.array([], dtype=np.float64)
    
    input_dict2 = {
        "x": input3,
        "y": input4,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input5 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input6 = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    out3 = np.array([], dtype=np.float16)
    
    input_dict3 = {
        "x": input5,
        "y": input6,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input8 = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    out4 = np.array([], dtype=np.float32)
    
    input_dict4 = {
        "x": input7,
        "y": input8,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input9 = np.array([1.0], dtype=np.float64)
    input10 = np.array([0.1], dtype=np.float64)
    out5 = np.array([], dtype=np.float64)

    input_dict5 = {
        "x": input9,
        "y": input10,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input11 = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    input12 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out6 = np.array([], dtype=np.float32)

    input_dict6 = {
        "x": input11,
        "y": input12,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input13 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input14 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out7 = np.array([], dtype=np.float32)

    input_dict7 = {
        "x": input13,
        "y": input14,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.special.xlogy"] = xlogy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.xlogy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.xlogy'.")


check_valid('torch.special.xlogy', generated_inputs['torch.special.xlogy'], lib="torch", suffix=0)
