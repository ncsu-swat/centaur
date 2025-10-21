
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def tril_indices_inputs():
    list_of_inputs = []
    
    input1 = np.int32(5)
    input2 = np.int32(5)
    input3 = np.int32(0)
    input4 = np.dtype(np.int64)
    input5 = "cpu"
    input6 = True
    
    input_dict1 = {
        "row": input1,
        "col": input2,
        "offset": input3,
        "dtype": input4,
        "device": input5,
        "pin_memory": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input1 = np.int32(3)
    input2 = np.int32(4)
    input3 = np.int32(-1)
    input4 = np.dtype(np.int32)
    input5 = "cpu"
    input6 = False
    
    input_dict2 = {
        "row": input1,
        "col": input2,
        "offset": input3,
        "dtype": input4,
        "device": input5,
        "pin_memory": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    

    return list_of_inputs

generated_inputs["torch.tril_indices"] = tril_indices_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.tril_indices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.tril_indices'.")


check_valid('torch.tril_indices', generated_inputs['torch.tril_indices'], lib="torch", suffix=0)
