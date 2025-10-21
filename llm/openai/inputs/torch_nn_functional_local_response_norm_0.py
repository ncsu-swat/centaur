
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def local_response_norm_inputs():
    list_of_inputs = []

    input1 = np.random.rand(3, 3, 3).astype(np.float32)
    size1 = 5
    alpha1 = 0.1
    beta1 = 0.5
    k1 = 2.0
    input_dict1 = {
        "input": input1,
        "size": size1,
        "alpha": alpha1,
        "beta": beta1,
        "k": k1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(2, 3, 4).astype(np.float32)
    size2 = 3
    alpha2 = 1.0
    beta2 = 0.0
    k2 = 1.0
    input_dict2 = {
        "input": input2,
        "size": size2,
        "alpha": alpha2,
        "beta": beta2,
        "k": k2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(1, 1, 1, 1).astype(np.float32)
    size3 = 1
    alpha3 = 0.5
    beta3 = 1.0
    k3 = 3.0
    input_dict3 = {
        "input": input3,
        "size": size3,
        "alpha": alpha3,
        "beta": beta3,
        "k": k3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(4, 4, 4).astype(np.float32)
    size4 = 2
    alpha4 = 0.2
    beta4 = 0.8
    k4 = 0.5
    input_dict4 = {
        "input": input4,
        "size": size4,
        "alpha": alpha4,
        "beta": beta4,
        "k": k4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    size5 = 4
    alpha5 = 1.5
    beta5 = 0.2
    k5 = 1.5
    input_dict5 = {
        "input": input5,
        "size": size5,
        "alpha": alpha5,
        "beta": beta5,
        "k": k5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.functional.local_response_norm"] = local_response_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.local_response_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.local_response_norm'.")


check_valid('torch.nn.functional.local_response_norm', generated_inputs['torch.nn.functional.local_response_norm'], lib="torch", suffix=0)
