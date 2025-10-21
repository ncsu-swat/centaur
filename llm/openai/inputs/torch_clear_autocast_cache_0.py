
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def clear_autocast_cache_inputs():
    list_of_inputs = []

    
    list_of_inputs.append({})
    
    list_of_inputs.append({})

    list_of_inputs.append({})

    list_of_inputs.append({})

    list_of_inputs.append({})
    
    list_of_inputs.append({})
    
    list_of_inputs.append({})

    list_of_inputs.append({})

    list_of_inputs.append({})

    list_of_inputs.append({})
    
    return list_of_inputs

generated_inputs["torch.clear_autocast_cache"] = clear_autocast_cache_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.clear_autocast_cache' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clear_autocast_cache'.")


check_valid('torch.clear_autocast_cache', generated_inputs['torch.clear_autocast_cache'], lib="torch", suffix=0)
