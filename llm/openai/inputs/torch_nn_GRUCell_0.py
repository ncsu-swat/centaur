
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def grucell_inputs():
    list_of_inputs = []
    input_size = 10
    hidden_size = 20

    input1 = np.random.rand(3, input_size).astype(np.float32)
    hidden1 = np.random.rand(3, hidden_size).astype(np.float32)
    input_dict1 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "dtype": torch.float32,
        "input": input1,
        "hidden": hidden1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(4, input_size).astype(np.float64)
    hidden2 = np.zeros((4, hidden_size), dtype=np.float64)
    input_dict2 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "dtype": torch.float64,
        "input": input2,
        "hidden": hidden2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(2, input_size).astype(np.float16)
    hidden3 = np.random.rand(2, hidden_size).astype(np.float16)
    input_dict3 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "dtype": torch.float16,
        "input": input3,
        "hidden": hidden3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(5, input_size).astype(np.float32)
    hidden4 = np.random.rand(5, hidden_size).astype(np.float32)
    input_dict4 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": False,
        "dtype": torch.float32,
        "input": input4,
        "hidden": hidden4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.rand(1, input_size).astype(np.float32)
    hidden5 = np.zeros((1, hidden_size), dtype=np.float32)
    input_dict5 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "dtype": torch.float32,
        "input": input5,
        "hidden": hidden5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.GRUCell"] = grucell_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.GRUCell' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GRUCell'.")


check_valid('torch.nn.GRUCell', generated_inputs['torch.nn.GRUCell'], lib="torch", suffix=0)
