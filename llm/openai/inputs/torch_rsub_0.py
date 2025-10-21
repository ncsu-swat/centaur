
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rsub_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    other1 = np.array([0.1, 0.2, 0.3])
    alpha1 = 2.0
    
    input_dict1 = {
        "input": input1,
        "other": other1,
        "alpha": alpha1
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.rsub"] = rsub_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.rsub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rsub'.")


check_valid('torch.rsub', generated_inputs['torch.rsub'], lib="torch", suffix=0)
