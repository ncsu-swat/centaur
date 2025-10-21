
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sparse_coo_tensor_inputs():
    list_of_inputs = []
    
    indices = np.array([[0, 1], [1, 2]])
    values = np.array([1, 2])
    size = (3, 3)
    dtype = torch.float32
    requires_grad = True
    
    input_dict = {
        "indices": torch.tensor(indices),
        "values": torch.tensor(values),
        "size": size,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["torch.sparse_coo_tensor_2"] = sparse_coo_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.sparse_coo_tensor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sparse_coo_tensor_2'.")


check_valid('torch.sparse_coo_tensor', generated_inputs['torch.sparse_coo_tensor_2'], lib="torch", suffix=2)
