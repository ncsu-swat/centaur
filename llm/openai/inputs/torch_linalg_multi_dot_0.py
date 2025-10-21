
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def multi_dot_inputs():
    list_of_inputs = []
    
    input1 = [np.random.rand(2, 3), np.random.rand(3, 4)]
    list_of_inputs.append(copy.deepcopy({"tensors": input1}))
    
    return list_of_inputs

generated_inputs["torch.linalg.multi_dot"] = multi_dot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.multi_dot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.multi_dot'.")


check_valid('torch.linalg.multi_dot', generated_inputs['torch.linalg.multi_dot'], lib="torch", suffix=0)
