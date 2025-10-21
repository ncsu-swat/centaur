
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sym_float_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0]).astype(np.float32)
    list_of_inputs.append({"a": input1})
    
    input2 = np.array([-1.0]).astype(np.float32)
    list_of_inputs.append({"a": input2})
    
    input3 = np.array([0.0]).astype(np.float32)
    list_of_inputs.append({"a": input3})
    
    return list_of_inputs

generated_inputs["torch.sym_float"] = sym_float_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sym_float' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sym_float'.")


check_valid('torch.sym_float', generated_inputs['torch.sym_float'], lib="torch", suffix=0)
