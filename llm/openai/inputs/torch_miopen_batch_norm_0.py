
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def miopen_batch_norm_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    weight1 = np.random.rand(4).astype(np.float32)
    bias1 = np.random.rand(4).astype(np.float32)
    running_mean1 = np.random.rand(4).astype(np.float32)
    running_var1 = np.random.rand(4).astype(np.float32)
    training1 = True
    exponential_average_factor1 = 0.1
    eps1 = 1e-5
    
    input_dict1 = {
        "input": input1,
        "weight": weight1,
        "bias": bias1,
        "running_mean": running_mean1,
        "running_var": running_var1,
        "training": training1,
        "exponential_average_factor": exponential_average_factor1,
        "eps": eps1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.miopen_batch_norm"] = miopen_batch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.miopen_batch_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.miopen_batch_norm'.")


check_valid('torch.miopen_batch_norm', generated_inputs['torch.miopen_batch_norm'], lib="torch", suffix=0)
