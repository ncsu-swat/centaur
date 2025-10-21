
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def arange_inputs():
    list_of_inputs = []
    dtype = np.float64
    requires_grad = False

    input1 = {'start': 0.0, 'end': 5.0, 'step': 1.0, 'out': torch.empty(0, dtype=torch.float64), 'dtype': dtype, 'requires_grad': requires_grad}
    list_of_inputs.append(copy.deepcopy(input1))

    return list_of_inputs

generated_inputs["torch.arange"] = arange_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.arange' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.arange'.")


check_valid('torch.arange', generated_inputs['torch.arange'], lib="torch", suffix=0)
