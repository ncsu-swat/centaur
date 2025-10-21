
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rrelu_inputs():
    list_of_inputs = []
    
    input1 = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    lower1 = 0.1
    upper1 = 0.3
    training1 = False
    inplace1 = False

    input_dict1 = {
        "input": input1,
        "lower": lower1,
        "upper": upper1,
        "training": training1,
        "inplace": inplace1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    lower2 = 0.05
    upper2 = 0.2
    training2 = True
    inplace2 = True

    input_dict2 = {
        "input": input2,
        "lower": lower2,
        "upper": upper2,
        "training": training2,
        "inplace": inplace2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([-2.5, -1.0, 0.5, 1.5, 2.0], dtype=np.float32)
    lower3 = 1.0 / 8
    upper3 = 1.0 / 3
    training3 = False
    inplace3 = False

    input_dict3 = {
        "input": input3,
        "lower": lower3,
        "upper": upper3,
        "training": training3,
        "inplace": inplace3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower4 = 0.25
    upper4 = 0.4
    training4 = True
    inplace4 = False

    input_dict4 = {
        "input": input4,
        "lower": lower4,
        "upper": upper4,
        "training": training4,
        "inplace": inplace4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([], dtype=np.float32)
    lower5 = 1.0 / 8
    upper5 = 1.0 / 3
    training5 = False
    inplace5 = False

    input_dict5 = {
        "input": input5,
        "lower": lower5,
        "upper": upper5,
        "training": training5,
        "inplace": inplace5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    lower6 = 0.1
    upper6 = 0.3
    training6 = True
    inplace6 = True

    input_dict6 = {
        "input": input6,
        "lower": lower6,
        "upper": upper6,
        "training": training6,
        "inplace": inplace6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lower7 = 1.0 / 8
    upper7 = 1.0 / 3
    training7 = False
    inplace7 = False

    input_dict7 = {
        "input": input7,
        "lower": lower7,
        "upper": upper7,
        "training": training7,
        "inplace": inplace7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    lower8 = 0.01
    upper8 = 0.02
    training8 = True
    inplace8 = True
    
    input_dict8 = {
        "input": input8,
        "lower": lower8,
        "upper": upper8,
        "training": training8,
        "inplace": inplace8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
    lower9 = 0.2
    upper9 = 0.4
    training9 = False
    inplace9 = True
    
    input_dict9 = {
        "input": input9,
        "lower": lower9,
        "upper": upper9,
        "training": training9,
        "inplace": inplace9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    lower10 = 1.0 / 4
    upper10 = 1.0 / 2
    training10 = True
    inplace10 = False

    input_dict10 = {
        "input": input10,
        "lower": lower10,
        "upper": upper10,
        "training": training10,
        "inplace": inplace10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.rrelu"] = rrelu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.rrelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.rrelu'.")


check_valid('torch.nn.functional.rrelu', generated_inputs['torch.nn.functional.rrelu'], lib="torch", suffix=0)
