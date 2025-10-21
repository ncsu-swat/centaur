
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def gather_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1, 2], [3, 4]])
    dim1 = 1
    index1 = np.array([[0, 0], [1, 0]])
    sparse_grad1 = False
    out1 = np.empty_like(index1)
    
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "index": index1,
        "sparse_grad": sparse_grad1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.gather"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.gather' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.gather'.")


check_valid('torch.gather', generated_inputs['torch.gather'], lib="torch", suffix=0)
