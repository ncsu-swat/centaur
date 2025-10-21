
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def trapz_inputs():
    list_of_inputs = []
    
    y1 = np.array([1.0, 2.0, 3.0])
    x1 = np.array([0.0, 1.0, 2.0])
    dim1 = -1
    
    input_dict1 = {
        "y": y1,
        "x": x1,
        "dim": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    y2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    x2 = np.array([[0.0, 1.0], [2.0, 3.0]])
    dim2 = 1
    
    input_dict2 = {
        "y": y2,
        "x": x2,
        "dim": dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    y3 = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
    x3 = np.array([0.0, 1.0, 2.0])
    dim3 = -1
    
    input_dict3 = {
        "y": y3,
        "x": x3,
        "dim": dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    y4 = np.array([1.0, 2.0, 3.0, 4.0])
    x4 = np.array([0.0, 0.5, 1.0, 1.5])
    dim4 = 0
    
    input_dict4 = {
        "y": y4,
        "x": x4,
        "dim": dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    y5 = np.array([1.0, 2.0, 3.0])
    x5 = np.array([0.0, 1.0, 2.0])
    dim5 = -1
    
    input_dict5 = {
        "y": y5,
        "x": x5,
        "dim": dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    y6 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x6 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    dim6 = 1
    
    input_dict6 = {
        "y": y6,
        "x": x6,
        "dim": dim6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    y7 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    x7 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dim7 = -1
    
    input_dict7 = {
        "y": y7,
        "x": x7,
        "dim": dim7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    y8 = np.array([[1.0, -2.0], [-3.0, 4.0]])
    x8 = np.array([[0.0, 1.0], [2.0, 3.0]])
    dim8 = 0
    
    input_dict8 = {
        "y": y8,
        "x": x8,
        "dim": dim8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    y9 = np.array([5.0, 6.0, 7.0, 8.0])
    x9 = np.array([10.0, 11.0, 12.0, 13.0])
    dim9 = -1

    input_dict9 = {
        "y": y9,
        "x": x9,
        "dim": dim9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    y10 = np.array([1.5, 2.5, 3.5])
    x10 = np.array([0.2, 0.4, 0.6])
    dim10 = -1

    input_dict10 = {
        "y": y10,
        "x": x10,
        "dim": dim10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.trapz"] = trapz_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.trapz' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.trapz'.")


check_valid('torch.trapz', generated_inputs['torch.trapz'], lib="torch", suffix=0)
