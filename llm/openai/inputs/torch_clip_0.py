
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clip_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    min1 = 0.0
    max1 = 2.0
    out1 = np.array([])

    input_dict1 = {
        "input": input1,
        "min": min1,
        "max": max1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([-1.0, 0.0, 1.0])
    min2 = -0.5
    max2 = 0.5
    out2 = np.array([])

    input_dict2 = {
        "input": input2,
        "min": min2,
        "max": max2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    min3 = 1.5
    max3 = 3.5
    out3 = np.array([])

    input_dict3 = {
        "input": input3,
        "min": min3,
        "max": max3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    min4 = -3.0
    max4 = -1.0
    out4 = np.array([])

    input_dict4 = {
        "input": input4,
        "min": min4,
        "max": max4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 5.0, 2.0, 8.0])
    min5 = 2.0
    max5 = 6.0
    out5 = np.array([])

    input_dict5 = {
        "input": input5,
        "min": min5,
        "max": max5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([0.1, 0.9, 0.5, 0.2])
    min6 = 0.2
    max6 = 0.8
    out6 = np.array([])

    input_dict6 = {
        "input": input6,
        "min": min6,
        "max": max6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    min7 = 3.0
    max7 = 6.0
    out7 = np.array([])

    input_dict7 = {
        "input": input7,
        "min": min7,
        "max": max7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([10.0, -5.0, 2.5, -1.0])
    min8 = -2.0
    max8 = 5.0
    out8 = np.array([])

    input_dict8 = {
        "input": input8,
        "min": min8,
        "max": max8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.5, 2.5, 3.5])
    min9 = 1.0
    max9 = 3.0
    out9 = np.array([])

    input_dict9 = {
        "input": input9,
        "min": min9,
        "max": max9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    min10 = 1.0
    max10 = 4.0
    out10 = np.array([])

    input_dict10 = {
        "input": input10,
        "min": min10,
        "max": max10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    

    return list_of_inputs

generated_inputs["torch.clip"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.clip' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clip'.")


check_valid('torch.clip', generated_inputs['torch.clip'], lib="torch", suffix=0)
