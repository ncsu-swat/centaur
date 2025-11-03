
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lu_unpack_inputs():
    list_of_inputs = []
    
    lu, pivots = torch.linalg.lu_factor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    input1 = lu.numpy()
    input2 = pivots.numpy()
    input_dict1 = {'LU_data': input1, 'LU_pivots': input2, 'unpack_data': True, 'unpack_pivots': True}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    lu, pivots = torch.linalg.lu_factor(torch.randn(3, 3))
    input3 = lu.numpy()
    input4 = pivots.numpy()
    input_dict2 = {'LU_data': input3, 'LU_pivots': input4, 'unpack_data': False, 'unpack_pivots': False}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    lu, pivots = torch.linalg.lu_factor(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    input5 = lu.numpy()
    input6 = pivots.numpy()
    input_dict3 = {'LU_data': input5, 'LU_pivots': input6, 'unpack_data': True, 'unpack_pivots': True}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    return list_of_inputs

generated_inputs["torch.lu_unpack"] = lu_unpack_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.lu_unpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lu_unpack'.")


check_valid('torch.lu_unpack', generated_inputs['torch.lu_unpack'], lib="torch", suffix=0)
