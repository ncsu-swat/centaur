
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def native_dropout_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    p1 = 0.5
    training1 = True
    input_dict1 = {
        "input": input1,
        "p": p1,
        "training": training1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    p2 = 0.2
    training2 = False
    input_dict2 = {
        "input": input2,
        "p": p2,
        "training": training2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    p3 = 0.8
    training3 = True
    input_dict3 = {
        "input": input3,
        "p": p3,
        "training": training3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([], dtype=np.float32)
    p4 = 0.0
    training4 = False
    input_dict4 = {
        "input": input4,
        "p": p4,
        "training": training4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float16)
    p5 = 0.75
    training5 = True
    input_dict5 = {
        "input": input5,
        "p": p5,
        "training": training5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    p6 = 0.1
    training6 = False
    input_dict6 = {
        "input": input6,
        "p": p6,
        "training": training6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([1.0, 2.0], dtype=np.float32)
    p7 = 0.9
    training7 = True
    input_dict7 = {
        "input": input7,
        "p": p7,
        "training": training7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    p8 = 0.3
    training8 = False
    input_dict8 = {
        "input": input8,
        "p": p8,
        "training": training8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0], dtype=np.float64)
    p9 = 0.6
    training9 = True
    input_dict9 = {
        "input": input9,
        "p": p9,
        "training": training9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    p10 = 0.4
    training10 = False
    input_dict10 = {
        "input": input10,
        "p": p10,
        "training": training10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.native_dropout"] = native_dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.native_dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.native_dropout'.")


check_valid('torch.native_dropout', generated_inputs['torch.native_dropout'], lib="torch", suffix=0)
