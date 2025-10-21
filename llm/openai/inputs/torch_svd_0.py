
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def svd_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(5, 3).astype(np.float32)
    some1 = True
    compute_uv1 = True
    out1 = (np.random.rand(5,3).astype(np.float32), np.random.rand(5,3).astype(np.float32), np.random.rand(5,3).astype(np.float32))
    input_dict1 = {"input": input1, "some": some1, "compute_uv": compute_uv1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.svd"] = svd_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.svd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.svd'.")


check_valid('torch.svd', generated_inputs['torch.svd'], lib="torch", suffix=0)
