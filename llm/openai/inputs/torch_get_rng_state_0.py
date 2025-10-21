
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def get_rng_state_inputs():
    list_of_inputs = []
    
    input1 = {}
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {}
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {}
    list_of_inputs.append(copy.deepcopy(input3))

    input4 = {}
    list_of_inputs.append(copy.deepcopy(input4))
    
    input5 = {}
    list_of_inputs.append(copy.deepcopy(input5))
    
    input6 = {}
    list_of_inputs.append(copy.deepcopy(input6))

    input7 = {}
    list_of_inputs.append(copy.deepcopy(input7))

    input8 = {}
    list_of_inputs.append(copy.deepcopy(input8))

    input9 = {}
    list_of_inputs.append(copy.deepcopy(input9))
    
    input10 = {}
    list_of_inputs.append(copy.deepcopy(input10))

    input11 = {}
    list_of_inputs.append(copy.deepcopy(input11))

    return list_of_inputs

generated_inputs["torch.get_rng_state"] = get_rng_state_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.get_rng_state' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.get_rng_state'.")


check_valid('torch.get_rng_state', generated_inputs['torch.get_rng_state'], lib="torch", suffix=0)
