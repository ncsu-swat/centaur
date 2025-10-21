
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rfftn_inputs():
    list_of_inputs = []

    input1 = np.random.rand(4, 4)
    s1 = (4, 4)
    dim1 = (0, 1)
    norm1 = "backward"
    out1 = np.zeros((4, 3), dtype=np.complex128)

    input_dict1 = {
        "input": input1,
        "s": s1,
        "dim": dim1,
        "norm": norm1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.fft.rfftn"] = rfftn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.rfftn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.rfftn'.")


check_valid('torch.fft.rfftn', generated_inputs['torch.fft.rfftn'], lib="torch", suffix=0)
