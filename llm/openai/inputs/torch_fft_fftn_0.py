
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def fftn_inputs():
    list_of_inputs = []

    input1 = np.random.rand(4, 4).astype(np.complex64)
    s1 = (4, 4)
    dim1 = (0, 1)
    norm1 = "forward"
    out1 = np.zeros_like(input1)

    input_dict1 = {
        "input": input1,
        "s": s1,
        "dim": dim1,
        "norm": norm1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.fft.fftn"] = fftn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.fftn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.fftn'.")


check_valid('torch.fft.fftn', generated_inputs['torch.fft.fftn'], lib="torch", suffix=0)
