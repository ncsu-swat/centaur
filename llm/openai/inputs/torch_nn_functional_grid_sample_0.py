
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def grid_sample_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(1, 3, 32, 32).astype(np.float32)
    grid1 = np.random.rand(1, 2, 32, 32).astype(np.float32)
    
    input_dict1 = {
        "input": input1,
        "grid": grid1,
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.grid_sample"] = grid_sample_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.grid_sample' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.grid_sample'.")


check_valid('torch.nn.functional.grid_sample', generated_inputs['torch.nn.functional.grid_sample'], lib="torch", suffix=0)
