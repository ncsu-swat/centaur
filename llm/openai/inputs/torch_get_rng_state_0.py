
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def get_rng_state_inputs():
    list_of_inputs = []

    input1 = torch.tensor(1).numpy()
    list_of_inputs.append({"torch.get_rng_state": input1})

    input2 = torch.tensor(0).numpy()
    list_of_inputs.append({"torch.get_rng_state": input2})

    input3 = torch.tensor(-1).numpy()
    list_of_inputs.append({"torch.get_rng_state": input3})

    input4 = torch.tensor(100).numpy()
    list_of_inputs.append({"torch.get_rng_state": input4})

    input5 = torch.tensor([1, 2, 3]).numpy()
    list_of_inputs.append({"torch.get_rng_state": input5})

    input6 = torch.tensor([0, 0, 0]).numpy()
    list_of_inputs.append({"torch.get_rng_state": input6})
    
    input7 = torch.tensor([-1, -2, -3]).numpy()
    list_of_inputs.append({"torch.get_rng_state": input7})
    
    input8 = torch.randn(2, 3).numpy()
    list_of_inputs.append({"torch.get_rng_state": input8})
    
    input9 = torch.zeros(3, 4, 5).numpy()
    list_of_inputs.append({"torch.get_rng_state": input9})

    input10 = torch.ones((5,)).numpy()
    list_of_inputs.append({"torch.get_rng_state": input10})

    input11 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    list_of_inputs.append({"torch.get_rng_state": input11})

    return list_of_inputs

generated_inputs["torch.get_rng_state"] = get_rng_state_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.get_rng_state' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.get_rng_state'.")


check_valid('torch.get_rng_state', generated_inputs['torch.get_rng_state'], lib="torch", suffix=0)
