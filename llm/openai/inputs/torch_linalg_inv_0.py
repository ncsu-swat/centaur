
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def linalg_inv_inputs():
    list_of_inputs = []
    
    A1 = np.random.rand(4, 4).astype(np.float32)
    out1 = np.zeros((4, 4)).astype(np.float32)
    input_dict1 = {'A': A1, 'out': out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    A2 = np.random.rand(2, 3, 4, 4).astype(np.float64)
    out2 = np.zeros((2, 3, 4, 4)).astype(np.float64)
    input_dict2 = {'A': A2, 'out': out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    A3 = np.random.rand(4, 4).astype(np.complex128)
    out3 = np.zeros((4, 4)).astype(np.complex128)
    input_dict3 = {'A': A3, 'out': out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    A5 = np.random.rand(5, 5).astype(np.float32) * -1
    out5 = np.zeros((5, 5)).astype(np.float32)
    input_dict5 = {'A': A5, 'out': out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    A6 = np.random.rand(2, 2).astype(np.float64) + 1j * np.random.rand(2, 2).astype(np.float64)
    out6 = np.zeros((2, 2)).astype(np.complex128)
    input_dict6 = {'A': A6, 'out': out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    A7 = np.random.rand(1, 1).astype(np.float32)
    out7 = np.zeros((1, 1)).astype(np.float32)
    input_dict7 = {'A': A7, 'out': out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    A8 = np.random.rand(6, 6).astype(np.float64)
    out8 = np.zeros((6, 6)).astype(np.float64)
    input_dict8 = {'A': A8, 'out': out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.linalg.inv"] = linalg_inv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.inv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.inv'.")


check_valid('torch.linalg.inv', generated_inputs['torch.linalg.inv'], lib="torch", suffix=0)
