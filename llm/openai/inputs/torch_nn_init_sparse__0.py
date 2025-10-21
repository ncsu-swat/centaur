
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sparse_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3).astype(np.float32)
    sparsity1 = 0.5
    input_dict1 = {"tensor": input1, "sparsity": sparsity1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(2, 2).astype(np.float64)
    sparsity2 = 0.2
    input_dict2 = {"tensor": input2, "sparsity": sparsity2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(2, 5).astype(np.float16)
    sparsity3 = 0.8
    input_dict3 = {"tensor": input3, "sparsity": sparsity3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 3).astype(np.float32)
    sparsity4 = 0.0
    input_dict4 = {"tensor": input4, "sparsity": sparsity4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(2, 2).astype(np.float64)
    sparsity5 = 1.0
    input_dict5 = {"tensor": input5, "sparsity": sparsity5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.nn.init.sparse_"] = sparse_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.sparse_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.sparse_'.")


check_valid('torch.nn.init.sparse_', generated_inputs['torch.nn.init.sparse_'], lib="torch", suffix=0)
