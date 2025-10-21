
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def unique_inputs():
    list_of_inputs = []

    input1 = np.array([1, 3, 2, 3], dtype=np.int64)
    input_dict1 = {
        "input": input1,
        "sorted": True,
        "return_inverse": False,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([1, 3, 2, 3], dtype=np.int64)
    input_dict2 = {
        "input": input2,
        "sorted": True,
        "return_inverse": True,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1, 3], [2, 3]], dtype=np.int64)
    input_dict3 = {
        "input": input3,
        "sorted": True,
        "return_inverse": True,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)
    input_dict4 = {
        "input": input4,
        "sorted": False,
        "return_inverse": False,
        "return_counts": True,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[-1, -3], [-2, -3]], dtype=np.int64)
    input_dict5 = {
        "input": input5,
        "sorted": False,
        "return_inverse": True,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]],
                       [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]],
                       [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]], dtype=np.int64)
    input_dict6 = {
        "input": input6,
        "sorted": True,
        "return_inverse": False,
        "return_counts": False,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]],
                       [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]],
                       [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]], dtype=np.int64)
    input_dict7 = {
        "input": input7,
        "sorted": False,
        "return_inverse": True,
        "return_counts": False,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]],
                       [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]],
                       [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]], dtype=np.int64)
    input_dict8 = {
        "input": input8,
        "sorted": True,
        "return_inverse": False,
        "return_counts": True,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1, 2, 2, 3, 3, 3], dtype=np.int64)
    input_dict9 = {
        "input": input9,
        "sorted": False,
        "return_inverse": False,
        "return_counts": True,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[1, 2], [1, 2]], dtype=np.int64)
    input_dict10 = {
        "input": input10,
        "sorted": True,
        "return_inverse": True,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.unique"] = unique_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.unique'.")


check_valid('torch.unique', generated_inputs['torch.unique'], lib="torch", suffix=0)
