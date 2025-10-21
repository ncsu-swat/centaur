
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def from_dlpack_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3]).astype(np.int32)
    list_of_inputs.append({"x": input1})
    
    input2 = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    list_of_inputs.append({"x": input2})
    
    return list_of_inputs

generated_inputs["torch.from_dlpack"] = from_dlpack_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.from_dlpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.from_dlpack'.")


check_valid('torch.from_dlpack', generated_inputs['torch.from_dlpack'], lib="torch", suffix=0)
