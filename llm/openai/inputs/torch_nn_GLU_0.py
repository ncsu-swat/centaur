
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def glu_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(4, 2).astype(np.float32)
    dim1 = -1
    input_dict1 = {"dim": dim1, "input": torch.tensor(input1)}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(2, 4).astype(np.float32)
    dim2 = 1
    input_dict2 = {"dim": dim2, "input": torch.tensor(input2)}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    return list_of_inputs

generated_inputs["torch.nn.GLU"] = glu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.GLU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GLU'.")


check_valid('torch.nn.GLU', generated_inputs['torch.nn.GLU'], lib="torch", suffix=0)
