
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def allclose_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([1.0, 2.0, 3.0])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([10000., 1e-07])
    input2 = np.array([10000.1, 1e-08])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([10000., 1e-08])
    input2 = np.array([10000.1, 1e-09])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1.0, np.nan])
    input2 = np.array([1.0, np.nan])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1.0, np.nan])
    input2 = np.array([1.0, np.nan])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input2 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([[1e9, 2e9], [3e9, 4e9]])
    input2 = np.array([[1e9 + 1, 2e9 + 2], [3e9 + 3, 4e9 + 4]])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([0.0, 0.0, 0.0])
    input2 = np.array([0.0, 0.0, 0.0])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = np.array([1.0, -2.0, 3.0])
    input2 = np.array([1.0, -2.0, 3.0])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = np.array([1.0, 2.0, np.nan])
    input2 = np.array([1.0, 2.0, np.nan])
    input_dict = {
        "input": input1,
        "other": input2,
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.allclose"] = allclose_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.allclose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.allclose'.")


check_valid('torch.allclose', generated_inputs['torch.allclose'], lib="torch", suffix=0)
