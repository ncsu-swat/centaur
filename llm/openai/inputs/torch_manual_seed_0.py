
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def manual_seed_inputs():
    list_of_inputs = []
    input_dict = {}
    input_dict['seed'] = np.int32(0)
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input_dict['seed'] = np.int64(123)
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input_dict['seed'] = np.int16(-10)
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input_dict['seed'] = np.int8(255)
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input_dict['seed'] = np.int32(99999)
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input_dict['seed'] = np.int64(-123456789)
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input_dict['seed'] = np.int16(1000)
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input_dict['seed'] = np.int8(0)
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {}
    input_dict['seed'] = np.int32(42)
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {}
    input_dict['seed'] = np.int64(1)
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.manual_seed"] = manual_seed_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.manual_seed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.manual_seed'.")


check_valid('torch.manual_seed', generated_inputs['torch.manual_seed'], lib="torch", suffix=0)
