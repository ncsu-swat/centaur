
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cholesky_ex_inputs():
    list_of_inputs = []
    
    input1 = np.array([[4, 12, -16],
                       [12, 37, -43],
                       [-16, -43, 98]], dtype=np.float64)
    input_dict1 = {'A': input1, 'upper': True, 'check_errors': True}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[4, 12, -16],
                       [12, 37, -43],
                       [-16, -43, 98]], dtype=np.float32)
    input_dict2 = {'A': input2, 'upper': False, 'check_errors': False}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]], dtype=np.float64)
    input_dict3 = {'A': input3, 'upper': True, 'check_errors': True}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[2, 1],
                       [1, 2]], dtype=np.float32)
    input_dict4 = {'A': input4, 'upper': False, 'check_errors': False}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[9, -3, 6],
                       [-3, 9, -6],
                       [6, -6, 9]], dtype=np.float64)
    input_dict5 = {'A': input5, 'upper': True, 'check_errors': True}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[25, 15, -5],
                       [15, 18, 0],
                       [-5, 0, 11]], dtype=np.float32)
    input_dict6 = {'A': input6, 'upper': False, 'check_errors': False}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.linalg.cholesky_ex"] = cholesky_ex_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.cholesky_ex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.cholesky_ex'.")


check_valid('torch.linalg.cholesky_ex', generated_inputs['torch.linalg.cholesky_ex'], lib="torch", suffix=0)
