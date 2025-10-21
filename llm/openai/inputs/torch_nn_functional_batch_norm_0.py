
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def batch_norm_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    running_mean1 = np.zeros(4).astype(np.float32)
    running_var1 = np.ones(4).astype(np.float32)
    weight1 = np.array([0.1, 0.2, 0.3, 0.4]).astype(np.float32)
    bias1 = np.array([0.5, 0.6, 0.7, 0.8]).astype(np.float32)
    training1 = True
    momentum1 = 0.1
    eps1 = 1e-5
    
    input_dict1 = {
        "input": input1,
        "running_mean": running_mean1,
        "running_var": running_var1,
        "weight": weight1,
        "bias": bias1,
        "training": training1,
        "momentum": momentum1,
        "eps": eps1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(1, 3, 4).astype(np.float32)
    running_mean2 = np.zeros(4).astype(np.float32)
    running_var2 = np.ones(4).astype(np.float32)
    weight2 = np.array([0.1, 0.2, 0.3, 0.4]).astype(np.float32)
    bias2 = np.array([0.5, 0.6, 0.7, 0.8]).astype(np.float32)
    training2 = False
    momentum2 = 0.1
    eps2 = 1e-5
    
    input_dict2 = {
        "input": input2,
        "running_mean": running_mean2,
        "running_var": running_var2,
        "weight": weight2,
        "bias": bias2,
        "training": training2,
        "momentum": momentum2,
        "eps": eps2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    
    return list_of_inputs

generated_inputs["torch.nn.functional.batch_norm"] = batch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.batch_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.batch_norm'.")


check_valid('torch.nn.functional.batch_norm', generated_inputs['torch.nn.functional.batch_norm'], lib="torch", suffix=0)
