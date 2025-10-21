
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def is_warn_always_enabled_inputs():
    list_of_inputs = []
    
    input1 = np.array(True)
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input1}))
    
    input2 = np.array(False)
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input2}))
    
    input3 = np.array([True])
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input3}))
    
    input4 = np.array([False])
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input4}))

    input5 = np.array([True, False])
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input5}))
    
    input6 = np.array([False, True])
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input6}))
    
    input7 = np.array([[True]])
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input7}))

    input8 = np.array([[False]])
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input8}))
    
    input9 = np.array([[True, False], [False, True]])
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input9}))

    input10 = np.array([[False, True], [True, False]])
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input10}))
    
    input11 = np.array([True, True, True])
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input11}))

    input12 = np.array([False, False, False])
    list_of_inputs.append(copy.deepcopy({"torch.is_warn_always_enabled": input12}))
    
    return list_of_inputs

generated_inputs["torch.is_warn_always_enabled"] = is_warn_always_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_warn_always_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_warn_always_enabled'.")


check_valid('torch.is_warn_always_enabled', generated_inputs['torch.is_warn_always_enabled'], lib="torch", suffix=0)
