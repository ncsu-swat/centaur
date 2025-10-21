
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def group_norm_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(4, 3, 28, 28).astype(np.float32)
    num_groups1 = 3
    weight1 = np.random.rand(3).astype(np.float32)
    bias1 = np.random.rand(3).astype(np.float32)
    eps1 = 1e-5
    
    input_dict1 = {
        "input": input1,
        "num_groups": num_groups1,
        "weight": weight1,
        "bias": bias1,
        "eps": eps1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(2, 5, 10, 10).astype(np.float32)
    num_groups2 = 5
    weight2 = np.random.rand(5).astype(np.float32)
    bias2 = np.random.rand(5).astype(np.float32)
    eps2 = 1e-4
    
    input_dict2 = {
        "input": input2,
        "num_groups": num_groups2,
        "weight": weight2,
        "bias": bias2,
        "eps": eps2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.nn.functional.group_norm"] = group_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.group_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.group_norm'.")


check_valid('torch.nn.functional.group_norm', generated_inputs['torch.nn.functional.group_norm'], lib="torch", suffix=0)
