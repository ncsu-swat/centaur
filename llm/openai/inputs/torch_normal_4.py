
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_normal_inputs():
    list_of_inputs = []
    
    input1 = {"mean": 0.5, "std": np.arange(1., 6.), "out": np.array([])}
    list_of_inputs.append(copy.deepcopy(input1))
    
    input2 = {"mean": np.arange(1., 6.), "std": 1.0, "out": np.array([])}
    list_of_inputs.append(copy.deepcopy(input2))

    return list_of_inputs

generated_inputs["torch.normal_4"] = torch_normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.normal_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.normal_4'.")


check_valid('torch.normal', generated_inputs['torch.normal_4'], lib="torch", suffix=4)
