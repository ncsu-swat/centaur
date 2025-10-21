
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def view_as_complex_copy_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    input_dict = {"torch.view_as_complex_copy": input1}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["torch.view_as_complex_copy"] = view_as_complex_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.view_as_complex_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.view_as_complex_copy'.")


check_valid('torch.view_as_complex_copy', generated_inputs['torch.view_as_complex_copy'], lib="torch", suffix=0)
