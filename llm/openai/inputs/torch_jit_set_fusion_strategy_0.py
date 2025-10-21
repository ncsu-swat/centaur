
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def set_fusion_strategy_inputs():
    list_of_inputs = []
    
    input1 = "default"
    list_of_inputs.append({"torch.jit.set_fusion_strategy": {"value": input1}})
    
    input2 = "true"
    list_of_inputs.append({"torch.jit.set_fusion_strategy": {"value": input2}})
    
    input3 = "false"
    list_of_inputs.append({"torch.jit.set_fusion_strategy": {"value": input3}})

    input4 = "separate"
    list_of_inputs.append({"torch.jit.set_fusion_strategy": {"value": input4}})
    
    input5 = "full"
    list_of_inputs.append({"torch.jit.set_fusion_strategy": {"value": input5}})
    
    return list_of_inputs

generated_inputs["torch.jit.set_fusion_strategy"] = set_fusion_strategy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.set_fusion_strategy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.set_fusion_strategy'.")


check_valid('torch.jit.set_fusion_strategy', generated_inputs['torch.jit.set_fusion_strategy'], lib="torch", suffix=0)
