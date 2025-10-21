
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_linalg_solve_inputs():
    list_of_inputs = []

    input1_A = np.random.rand(3, 3)
    input1_B = np.random.rand(3)
    input1_left = True
    input1_out = np.zeros((3,))

    input_dict1 = {
        'A': input1_A,
        'B': input1_B,
        'left': input1_left,
        'out': input1_out
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2_A = np.random.rand(2, 3, 3)
    input2_B = np.random.rand(2, 3, 3)
    input2_left = False
    input2_out = np.zeros((2, 3, 3))

    input_dict2 = {
        'A': input2_A,
        'B': input2_B,
        'left': input2_left,
        'out': input2_out
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3_A = np.random.rand(2, 2)
    input3_B = np.random.rand(2, 1)
    input3_left = True
    input3_out = np.zeros((2, 1))

    input_dict3 = {
        'A': input3_A,
        'B': input3_B,
        'left': input3_left,
        'out': input3_out
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.linalg.solve"] = torch_linalg_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.solve'.")


check_valid('torch.linalg.solve', generated_inputs['torch.linalg.solve'], lib="torch", suffix=0)
