
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def slogdet_inputs():
    list_of_inputs = []
    
    A1 = np.random.rand(3, 3).astype(np.float32)
    list_of_inputs.append({"A": A1, "out": None})
    
    return list_of_inputs

generated_inputs["torch.linalg.slogdet"] = slogdet_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.slogdet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.slogdet'.")


check_valid('torch.linalg.slogdet', generated_inputs['torch.linalg.slogdet'], lib="torch", suffix=0)
