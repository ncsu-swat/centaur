
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def alpha_dropout_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(5, 5).astype(np.float32)
    p1 = 0.2
    training1 = True
    inplace1 = False
    
    input_dict1 = {
        "input": input1,
        "p": p1,
        "training": training1,
        "inplace": inplace1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(10, 2, 3).astype(np.float32)
    p2 = 0.5
    training2 = False
    inplace2 = True
    
    input_dict2 = {
        "input": input2,
        "p": p2,
        "training": training2,
        "inplace": inplace2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(2).astype(np.float32)
    p3 = 0.0
    training3 = True
    inplace3 = False
    
    input_dict3 = {
        "input": input3,
        "p": p3,
        "training": training3,
        "inplace": inplace3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(3, 4, 5, 6).astype(np.float32)
    p4 = 1.0
    training4 = False
    inplace4 = True
    
    input_dict4 = {
        "input": input4,
        "p": p4,
        "training": training4,
        "inplace": inplace4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.rand(7, 7).astype(np.float32)
    p5 = 0.75
    training5 = True
    inplace5 = False
    
    input_dict5 = {
        "input": input5,
        "p": p5,
        "training": training5,
        "inplace": inplace5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(1).astype(np.float32)
    p6 = 0.3
    training6 = False
    inplace6 = True

    input_dict6 = {
        "input": input6,
        "p": p6,
        "training": training6,
        "inplace": inplace6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(4, 4).astype(np.float32)
    p7 = 0.6
    training7 = True
    inplace7 = False
    
    input_dict7 = {
        "input": input7,
        "p": p7,
        "training": training7,
        "inplace": inplace7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(2, 2, 2).astype(np.float32)
    p8 = 0.9
    training8 = False
    inplace8 = True

    input_dict8 = {
        "input": input8,
        "p": p8,
        "training": training8,
        "inplace": inplace8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.alpha_dropout"] = alpha_dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.alpha_dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.alpha_dropout'.")


check_valid('torch.nn.functional.alpha_dropout', generated_inputs['torch.nn.functional.alpha_dropout'], lib="torch", suffix=0)
