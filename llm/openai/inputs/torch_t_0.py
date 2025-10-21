
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def t_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor(0.1995)
    list_of_inputs.append({"torch.t": {"input": input1.numpy()}})
    
    return list_of_inputs

generated_inputs["torch.t"] = t_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.t' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.t'.")


check_valid('torch.t', generated_inputs['torch.t'], lib="torch", suffix=0)
