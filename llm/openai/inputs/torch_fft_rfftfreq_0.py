
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rfftfreq_inputs():
    list_of_inputs = []

    input_dict1 = {
        "n": 5,
        "d": 1.0,
        "out": torch.tensor([]),
        "dtype": torch.float32,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.fft.rfftfreq"] = rfftfreq_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.rfftfreq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.rfftfreq'.")


check_valid('torch.fft.rfftfreq', generated_inputs['torch.fft.rfftfreq'], lib="torch", suffix=0)
