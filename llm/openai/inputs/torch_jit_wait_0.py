
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def jit_wait_inputs():
    list_of_inputs = []
    future1 = [torch.jit.fork(lambda: torch.tensor(1.0))]
    list_of_inputs.append({"torch.jit.wait": future1})

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.jit.wait"] = jit_wait_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.wait' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.wait'.")


check_valid('torch.jit.wait', generated_inputs['torch.jit.wait'], lib="torch", suffix=0)
