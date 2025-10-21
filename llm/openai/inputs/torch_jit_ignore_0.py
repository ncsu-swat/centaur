
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_jit_ignore_inputs():
    list_of_inputs = []
    
    input1 = {'drop': True}
    list_of_inputs.append(copy.deepcopy(input1))
    
    input2 = {'drop': False}
    list_of_inputs.append(copy.deepcopy(input2))
    
    return list_of_inputs

generated_inputs["torch.jit.ignore"] = torch_jit_ignore_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.ignore' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.ignore'.")


check_valid('torch.jit.ignore', generated_inputs['torch.jit.ignore'], lib="torch", suffix=0)
