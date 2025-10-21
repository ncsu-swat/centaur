
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def addbmm_inputs():
    list_of_inputs = []
    input1 = np.random.rand(3, 5).astype(np.float32)
    batch1_1 = np.random.rand(10, 3, 4).astype(np.float32)
    batch2_1 = np.random.rand(10, 4, 5).astype(np.float32)
    beta_1 = 1.0
    alpha_1 = 1.0
    out_1 = np.zeros((3, 5)).astype(np.float32)

    input_dict1 = {
        "input": input1,
        "batch1": batch1_1,
        "batch2": batch2_1,
        "beta": beta_1,
        "alpha": alpha_1,
        "out": out_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(2, 2).astype(np.float32)
    batch1_2 = np.random.rand(5, 2, 3).astype(np.float32)
    batch2_2 = np.random.rand(5, 3, 2).astype(np.float32)
    beta_2 = 0.5
    alpha_2 = 2.0
    out_2 = np.zeros((2, 2)).astype(np.float32)

    input_dict2 = {
        "input": input2,
        "batch1": batch1_2,
        "batch2": batch2_2,
        "beta": beta_2,
        "alpha": alpha_2,
        "out": out_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    return list_of_inputs

generated_inputs["torch.addbmm"] = addbmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.addbmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addbmm'.")


check_valid('torch.addbmm', generated_inputs['torch.addbmm'], lib="torch", suffix=0)
