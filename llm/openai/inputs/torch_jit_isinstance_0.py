
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np
from typing import List, Dict, Type

def torch_jit_isinstance_inputs():
    list_of_inputs = []

    input1 = [1, 2, 3]
    target_type1 = List[int]
    list_of_inputs.append({"obj": input1, "target_type": target_type1})

    return list_of_inputs

generated_inputs["torch.jit.isinstance"] = torch_jit_isinstance_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.isinstance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.isinstance'.")


check_valid('torch.jit.isinstance', generated_inputs['torch.jit.isinstance'], lib="torch", suffix=0)
