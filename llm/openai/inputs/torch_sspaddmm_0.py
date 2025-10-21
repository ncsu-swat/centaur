
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sspaddmm_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mat1_1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    mat2_1 = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    beta_1 = 0.5
    alpha_1 = 2.0
    out_1 = np.zeros((3, 2), dtype=np.float32)

    input_dict1 = {
        "input": torch.sparse_coo_tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), input1, (2, 3)),
        "mat1": torch.sparse_coo_tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), mat1_1.flatten(), (2, 2)),
        "mat2": mat2_1,
        "beta": beta_1,
        "alpha": alpha_1,
        "out": out_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.sspaddmm"] = sspaddmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sspaddmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sspaddmm'.")


check_valid('torch.sspaddmm', generated_inputs['torch.sspaddmm'], lib="torch", suffix=0)
