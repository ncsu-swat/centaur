
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def set_autocast_cpu_dtype_inputs():
    list_of_inputs = []
    
    dtype1 = np.dtype(np.float16)
    input_dict = {"dtype": dtype1}
    list_of_inputs.append(input_dict)
    
    dtype2 = np.dtype(np.float32)
    input_dict = {"dtype": dtype2}
    list_of_inputs.append(input_dict)
    
    dtype3 = np.dtype(np.float64)
    input_dict = {"dtype": dtype3}
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["torch.set_autocast_cpu_dtype"] = set_autocast_cpu_dtype_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.set_autocast_cpu_dtype' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_autocast_cpu_dtype'.")


check_valid('torch.set_autocast_cpu_dtype', generated_inputs['torch.set_autocast_cpu_dtype'], lib="torch", suffix=0)
