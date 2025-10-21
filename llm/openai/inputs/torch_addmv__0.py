
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def addmv_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    mat1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype(np.float32)
    vec1 = np.array([0.1, 0.2]).astype(np.float32)
    beta1 = 0.5
    alpha1 = 1.0
    input_dict1 = {"input": input1, "mat": mat1, "vec": vec1, "beta": beta1, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.addmv_"] = addmv_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.addmv_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmv_'.")


check_valid('torch.addmv_', generated_inputs['torch.addmv_'], lib="torch", suffix=0)
