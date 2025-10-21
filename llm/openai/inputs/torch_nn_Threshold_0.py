
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def threshold_inputs():
    list_of_inputs = []
    
    input1 = np.array([-1.0, 0.0, 1.0, 2.0])
    input_dict1 = {
        "threshold": 0.0,
        "value": 0.5,
        "inplace": False,
        "input": torch.tensor(input1).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([-3.0, -2.0, -1.0])
    input_dict2 = {
        "threshold": -2.0,
        "value": 1.0,
        "inplace": True,
        "input": torch.tensor(input2).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([1.0, 2.0, 3.0, 4.0]).reshape((2, 2))
    input_dict3 = {
        "threshold": 2.5,
        "value": -1.0,
        "inplace": False,
        "input": torch.tensor(input3).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([0.0, 0.0, 0.0])
    input_dict4 = {
        "threshold": 0.0,
        "value": 10.0,
        "inplace": True,
        "input": torch.tensor(input4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([-1.5, 0.5, 2.5, -0.5])
    input_dict5 = {
        "threshold": 0.0,
        "value": -2.0,
        "inplace": False,
        "input": torch.tensor(input5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0, 2.0, 3.0]).reshape((1, 3))
    input_dict6 = {
        "threshold": 1.5,
        "value": 0.0,
        "inplace": True,
        "input": torch.tensor(input6).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input_dict7 = {
        "threshold": 0.0,
        "value": 5.0,
        "inplace": False,
        "input": torch.tensor(input7).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([5.0, 5.0, 5.0])
    input_dict8 = {
        "threshold": 5.0,
        "value": 1.0,
        "inplace": True,
        "input": torch.tensor(input8).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([1.2, 3.4, 5.6, 7.8])
    input_dict9 = {
        "threshold": 4.0,
        "value": 2.0,
        "inplace": False,
        "input": torch.tensor(input9).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[-2.0, 1.0, 0.0], [3.0, -1.0, 2.0]])
    input_dict10 = {
        "threshold": -1.0,
        "value": 0.0,
        "inplace": True,
        "input": torch.tensor(input10).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.Threshold"] = threshold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Threshold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Threshold'.")


check_valid('torch.nn.Threshold', generated_inputs['torch.nn.Threshold'], lib="torch", suffix=0)
