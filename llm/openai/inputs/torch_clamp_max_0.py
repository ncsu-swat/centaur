
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clamp_max_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    max1 = 2.5
    out1 = np.array([], dtype=np.float32)
    
    input_dict1 = {
        "input": input1,
        "max": max1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    max2 = 1.0
    out2 = np.array([], dtype=np.float64)
    
    input_dict2 = {
        "input": input2,
        "max": max2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([5.0, 1.0, 8.0, 2.0], dtype=np.float16)
    max3 = 6.0
    out3 = np.array([], dtype=np.float16)
    
    input_dict3 = {
        "input": input3,
        "max": max3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    max4 = 5.0
    out4 = np.array([], dtype=np.float32)
    
    input_dict4 = {
        "input": input4,
        "max": max4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    max5 = 1.0
    out5 = np.array([], dtype=np.float32)
    
    input_dict5 = {
        "input": input5,
        "max": max5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    max6 = -1.5
    out6 = np.array([], dtype=np.float32)
    
    input_dict6 = {
        "input": input6,
        "max": max6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    max7 = 3.0
    out7 = np.array([], dtype=np.float32)
    
    input_dict7 = {
        "input": input7,
        "max": max7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([1.0], dtype=np.float64)
    max8 = 1.0
    out8 = np.array([], dtype=np.float64)
    
    input_dict8 = {
        "input": input8,
        "max": max8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    max9 = 4.5
    out9 = np.array([], dtype=np.float32)

    input_dict9 = {
        "input": input9,
        "max": max9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[-2.0, 1.0], [0.5, -3.0]], dtype=np.float32)
    max10 = 0.0
    out10 = np.array([], dtype=np.float32)

    input_dict10 = {
        "input": input10,
        "max": max10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.clamp_max"] = clamp_max_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.clamp_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clamp_max'.")


check_valid('torch.clamp_max', generated_inputs['torch.clamp_max'], lib="torch", suffix=0)
