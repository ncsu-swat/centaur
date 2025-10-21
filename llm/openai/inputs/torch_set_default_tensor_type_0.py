
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def set_default_tensor_type_inputs():
    list_of_inputs = []
    
    input1 = {"t": "torch.FloatTensor"}
    list_of_inputs.append(copy.deepcopy(input1))
    
    input2 = {"t": "torch.DoubleTensor"}
    list_of_inputs.append(copy.deepcopy(input2))
    
    input3 = {"t": "torch.HalfTensor"}
    list_of_inputs.append(copy.deepcopy(input3))
    
    return list_of_inputs

generated_inputs["torch.set_default_tensor_type"] = set_default_tensor_type_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.set_default_tensor_type' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_default_tensor_type'.")


check_valid('torch.set_default_tensor_type', generated_inputs['torch.set_default_tensor_type'], lib="torch", suffix=0)
