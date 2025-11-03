
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def isreal_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3])
    list_of_inputs.append({"input": input1})
    
    input2 = np.array([1+1j, 2+0j, 3-1j])
    list_of_inputs.append({"input": input2})
    
    input3 = np.array([[1, 2], [3, 4]])
    list_of_inputs.append({"input": input3})
    
    input4 = np.array([[1+1j, 2-1j], [3+0j, 4-0j]])
    list_of_inputs.append({"input": input4})
    
    input5 = np.array([1.0, 2.5, 3.7])
    list_of_inputs.append({"input": input5})
    
    input6 = np.array([1.0+0.0j, 2.5-1.2j, 3.7+0.5j])
    list_of_inputs.append({"input": input6})
    
    return list_of_inputs

generated_inputs["torch.isreal"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.isreal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isreal'.")


check_valid('torch.isreal', generated_inputs['torch.isreal'], lib="torch", suffix=0)
