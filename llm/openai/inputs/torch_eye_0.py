
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_eye_inputs():
    list_of_inputs = []

    input1 = {'n': 3, 'm': 4, 'out': torch.empty(3, 4), 'dtype': torch.float64, 'requires_grad': True}
    list_of_inputs.append(copy.deepcopy(input1))

    return list_of_inputs

generated_inputs["torch.eye"] = torch_eye_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.eye' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.eye'.")


check_valid('torch.eye', generated_inputs['torch.eye'], lib="torch", suffix=0)
