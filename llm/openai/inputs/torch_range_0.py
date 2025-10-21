
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_range_inputs():
    list_of_inputs = []
    
    input1 = {
        'end': 5.0,
        'start': 0.0,
        'step': 1.0,
        'out': np.array([]).astype(np.float32),
        'dtype': np.float32,
        'requires_grad': False
    }
    
    return list_of_inputs

generated_inputs["torch.range"] = torch_range_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.range' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.range'.")


check_valid('torch.range', generated_inputs['torch.range'], lib="torch", suffix=0)
