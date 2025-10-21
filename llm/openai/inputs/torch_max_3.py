
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_max_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3).astype(np.float32)
    dim1 = (0,)
    keepdim1 = True
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(3, 4, 5).astype(np.float32)
    dim2 = (1,)
    keepdim2 = False
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(2, 2).astype(np.float32)
    dim3 = ()
    keepdim3 = True
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "keepdim": keepdim3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.max_3"] = torch_max_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.max_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.max_3'.")


check_valid('torch.max', generated_inputs['torch.max_3'], lib="torch", suffix=3)
