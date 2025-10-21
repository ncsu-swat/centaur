
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_min_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    dim1 = 0
    keepdim1 = False
    out1 = None
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    dim2 = 1
    keepdim2 = True
    out2 = None
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.min_2"] = torch_min_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.min_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.min_2'.")


check_valid('torch.min', generated_inputs['torch.min_2'], lib="torch", suffix=2)
