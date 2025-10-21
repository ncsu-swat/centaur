
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def script_if_tracing_inputs():
    list_of_inputs = []
    
    fn1 = [torch.tensor([1, 2, 3]).numpy()]
    input_dict1 = {"fn": fn1, "alternative_fn": [torch.tensor([4,5,6]).numpy()]}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.jit.script_if_tracing"] = script_if_tracing_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.script_if_tracing' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.script_if_tracing'.")


check_valid('torch.jit.script_if_tracing', generated_inputs['torch.jit.script_if_tracing'], lib="torch", suffix=0)
