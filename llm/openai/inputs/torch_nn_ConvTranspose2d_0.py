
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def conv_transpose2d_inputs():
    list_of_inputs = []

    input1 = np.random.rand(20, 16, 50, 100).astype(np.float32)
    input_dict1 = {
        "in_channels": 16,
        "out_channels": 33,
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.float32,
        "input": input1,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.ConvTranspose2d"] = conv_transpose2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ConvTranspose2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConvTranspose2d'.")


check_valid('torch.nn.ConvTranspose2d', generated_inputs['torch.nn.ConvTranspose2d'], lib="torch", suffix=0)
