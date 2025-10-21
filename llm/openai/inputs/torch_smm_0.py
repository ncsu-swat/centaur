
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def smm_inputs():
    list_of_inputs = []
    
    input1 = torch.sparse_coo_tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([1, 2, 3, 4]), (2, 2)).numpy()
    mat1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    list_of_inputs.append({"input": input1, "mat": mat1})
    
    return list_of_inputs

generated_inputs["torch.smm"] = smm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.smm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.smm'.")


check_valid('torch.smm', generated_inputs['torch.smm'], lib="torch", suffix=0)
