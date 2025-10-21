
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def tensorinv_inputs():
    list_of_inputs = []
    
    A1 = np.eye(4).astype(np.float64)
    ind1 = 1
    out1 = torch.Tensor()
    input_dict1 = {'A': A1, 'ind': ind1, 'out': out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.linalg.tensorinv"] = tensorinv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.tensorinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.tensorinv'.")


check_valid('torch.linalg.tensorinv', generated_inputs['torch.linalg.tensorinv'], lib="torch", suffix=0)
