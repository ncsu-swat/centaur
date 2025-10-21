
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def histc_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 1.0])
    bins1 = 4
    min1 = 0.0
    max1 = 3.0
    out1 = np.zeros(4)
    
    input_dict1 = {
        "input": input1,
        "bins": bins1,
        "min": min1,
        "max": max1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, 0.0, 1.0, 2.0])
    bins2 = 5
    min2 = -2.0
    max2 = 3.0
    out2 = np.zeros(5)
    
    input_dict2 = {
        "input": input2,
        "bins": bins2,
        "min": min2,
        "max": max2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    bins3 = 6
    min3 = 1.0
    max3 = 6.0
    out3 = np.zeros(6)
    
    input_dict3 = {
        "input": input3,
        "bins": bins3,
        "min": min3,
        "max": max3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bins4 = 3
    min4 = 0.0
    max4 = 0.0
    out4 = np.zeros(3)

    input_dict4 = {
        "input": input4,
        "bins": bins4,
        "min": min4,
        "max": max4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[1.0, 2.0], [3.0, 4.0]])
    bins5 = 4
    min5 = 0.0
    max5 = 5.0
    out5 = np.zeros(4)
    
    input_dict5 = {
        "input": input5,
        "bins": bins5,
        "min": min5,
        "max": max5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, np.nan, 4.0])
    bins6 = 4
    min6 = 0.0
    max6 = 5.0
    out6 = np.zeros(4)

    input_dict6 = {
        "input": input6,
        "bins": bins6,
        "min": min6,
        "max": max6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0, 2.0, 3.0])
    bins7 = 2
    min7 = 1.5
    max7 = 2.5
    out7 = np.zeros(2)

    input_dict7 = {
        "input": input7,
        "bins": bins7,
        "min": min7,
        "max": max7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([1.0, 1.0, 1.0, 1.0])
    bins8 = 5
    min8 = 0.0
    max8 = 2.0
    out8 = np.zeros(5)

    input_dict8 = {
        "input": input8,
        "bins": bins8,
        "min": min8,
        "max": max8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
    bins9 = 5
    min9 = -6.0
    max9 = 0.0
    out9 = np.zeros(5)

    input_dict9 = {
        "input": input9,
        "bins": bins9,
        "min": min9,
        "max": max9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0.0, 0.0, 0.0])
    bins10 = 2
    min10 = 0.0
    max10 = 1.0
    out10 = np.zeros(2)
    
    input_dict10 = {
        "input": input10,
        "bins": bins10,
        "min": min10,
        "max": max10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.histc"] = histc_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.histc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.histc'.")


check_valid('torch.histc', generated_inputs['torch.histc'], lib="torch", suffix=0)
