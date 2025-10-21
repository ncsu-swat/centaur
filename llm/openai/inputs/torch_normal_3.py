
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_normal_inputs():
    list_of_inputs = []

    input1_mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input1_std = 0.5
    input1_out = np.empty((3,), dtype=np.float32)
    input_dict1 = {"mean": input1_mean, "std": input1_std, "out": input1_out}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.normal_3"] = torch_normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.normal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.normal_3'.")


check_valid('torch.normal', generated_inputs['torch.normal_3'], lib="torch", suffix=3)
