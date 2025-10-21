
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sgn_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    out1 = np.array([])
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([-1.0, -2.0, -3.0])
    out2 = np.array([])
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([0.0, 0.0, 0.0])
    out3 = np.array([])
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0 + 1j, 2.0 - 2j, 3.0 + 0j], dtype=complex128)
    out4 = np.array([])
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    

    return list_of_inputs

generated_inputs["torch.sgn"] = sgn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sgn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sgn'.")


check_valid('torch.sgn', generated_inputs['torch.sgn'], lib="torch", suffix=0)
