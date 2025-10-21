
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def nan_to_num_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, np.nan, 3.0, np.inf, -np.inf])
    nan1 = 0.0
    posinf1 = 1e10
    neginf1 = -1e10
    out1 = np.zeros_like(input1)
    
    input_dict1 = {
        "input": input1,
        "nan": nan1,
        "posinf": posinf1,
        "neginf": neginf1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, np.nan], [np.inf, -np.inf]])
    nan2 = 1.0
    posinf2 = 1e5
    neginf2 = -1e5
    out2 = np.zeros_like(input2)
    
    input_dict2 = {
        "input": input2,
        "nan": nan2,
        "posinf": posinf2,
        "neginf": neginf2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([np.nan, np.nan, np.nan])
    nan3 = 5.0
    posinf3 = 1e8
    neginf3 = -1e8
    out3 = np.zeros_like(input3)

    input_dict3 = {
        "input": input3,
        "nan": nan3,
        "posinf": posinf3,
        "neginf": neginf3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0, 2.0, 3.0])
    nan4 = 0.0
    posinf4 = 1e6
    neginf4 = -1e6
    out4 = np.zeros_like(input4)

    input_dict4 = {
        "input": input4,
        "nan": nan4,
        "posinf": posinf4,
        "neginf": neginf4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[-np.inf, 1.0], [2.0, np.nan]])
    nan5 = 2.0
    posinf5 = 1e7
    neginf5 = -1e7
    out5 = np.zeros_like(input5)

    input_dict5 = {
        "input": input5,
        "nan": nan5,
        "posinf": posinf5,
        "neginf": neginf5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([np.inf, -np.inf, 1.0])
    nan6 = 0.0
    posinf6 = 1e9
    neginf6 = -1e9
    out6 = np.zeros_like(input6)

    input_dict6 = {
        "input": input6,
        "nan": nan6,
        "posinf": posinf6,
        "neginf": neginf6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0, 2.0, np.nan, 4.0, np.inf])
    nan7 = 3.0
    posinf7 = 1e4
    neginf7 = -1e4
    out7 = np.zeros_like(input7)

    input_dict7 = {
        "input": input7,
        "nan": nan7,
        "posinf": posinf7,
        "neginf": neginf7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[np.nan, np.inf], [-np.inf, 1.0]])
    nan8 = 1.0
    posinf8 = 1e3
    neginf8 = -1e3
    out8 = np.zeros_like(input8)

    input_dict8 = {
        "input": input8,
        "nan": nan8,
        "posinf": posinf8,
        "neginf": neginf8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, -np.inf, np.nan, 3.0])
    nan9 = 2.0
    posinf9 = 1e2
    neginf9 = -1e2
    out9 = np.zeros_like(input9)

    input_dict9 = {
        "input": input9,
        "nan": nan9,
        "posinf": posinf9,
        "neginf": neginf9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([np.nan])
    nan10 = 0.0
    posinf10 = 1e1
    neginf10 = -1e1
    out10 = np.zeros_like(input10)

    input_dict10 = {
        "input": input10,
        "nan": nan10,
        "posinf": posinf10,
        "neginf": neginf10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nan_to_num"] = nan_to_num_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nan_to_num' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nan_to_num'.")


check_valid('torch.nan_to_num', generated_inputs['torch.nan_to_num'], lib="torch", suffix=0)
