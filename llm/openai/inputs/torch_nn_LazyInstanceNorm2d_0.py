
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lazyinstancenorm2d_inputs():
    list_of_inputs = []

    input1 = np.random.rand(1, 3, 224, 224).astype(np.float32)
    input_dict1 = {
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": np.float32,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(1, 1, 64, 64).astype(np.float32)
    input_dict2 = {
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": np.float32,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.nn.LazyInstanceNorm2d"] = lazyinstancenorm2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LazyInstanceNorm2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LazyInstanceNorm2d'.")


check_valid('torch.nn.LazyInstanceNorm2d', generated_inputs['torch.nn.LazyInstanceNorm2d'], lib="torch", suffix=0)
