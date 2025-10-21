
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def conv_transpose1d_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(1, 3, 5).astype(np.float32)
    weight1 = np.random.rand(3, 2, 3).astype(np.float32)
    bias1 = np.random.rand(2).astype(np.float32)
    stride1 = 2
    padding1 = 1
    output_padding1 = 0
    groups1 = 1
    dilation1 = 1
    
    input_dict1 = {
        "input": input1,
        "weight": weight1,
        "bias": bias1,
        "stride": stride1,
        "padding": padding1,
        "output_padding": output_padding1,
        "groups": groups1,
        "dilation": dilation1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(1, 6, 10).astype(np.float32)
    weight2 = np.random.rand(6, 4, 3).astype(np.float32)
    bias2 = np.random.rand(4).astype(np.float32)
    stride2 = 1
    padding2 = 2
    output_padding2 = 1
    groups2 = 1
    dilation2 = 2
    
    input_dict2 = {
        "input": input2,
        "weight": weight2,
        "bias": bias2,
        "stride": stride2,
        "padding": padding2,
        "output_padding": output_padding2,
        "groups": groups2,
        "dilation": dilation2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.conv_transpose1d"] = conv_transpose1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.conv_transpose1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv_transpose1d'.")


check_valid('torch.nn.functional.conv_transpose1d', generated_inputs['torch.nn.functional.conv_transpose1d'], lib="torch", suffix=0)
