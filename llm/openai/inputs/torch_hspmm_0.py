
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def hspmm_inputs():
    list_of_inputs = []

    input1 = torch.sparse_coo_tensor(np.array([[0, 1], [1, 0]]), np.array([1, 2]), (2, 2)).to_dense().numpy()
    input2 = torch.randn(2, 2).numpy()
    out1 = torch.tensor(np.array([])).numpy()
    list_of_inputs.append({"mat1": input1, "mat2": input2, "out": out1})

    input1 = torch.sparse_coo_tensor(np.array([[0, 0], [0, 1]]), np.array([3, 4]), (2, 2)).to_dense().numpy()
    input2 = torch.randn(2, 2).numpy()
    out1 = torch.tensor(np.array([])).numpy()
    list_of_inputs.append({"mat1": input1, "mat2": input2, "out": out1})

    return list_of_inputs

generated_inputs["torch.hspmm"] = hspmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.hspmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hspmm'.")


check_valid('torch.hspmm', generated_inputs['torch.hspmm'], lib="torch", suffix=0)
