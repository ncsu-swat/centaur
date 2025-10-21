
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def empty_strided_inputs():
    list_of_inputs = []
    
    input1 = (2,)
    stride1 = (1,)
    dtype1 = torch.float32
    pin_memory1 = False
    requires_grad1 = False
    
    input_dict1 = {
        "size": input1,
        "stride": stride1,
        "dtype": dtype1,
        "pin_memory": pin_memory1,
        "requires_grad": requires_grad1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = (3, 4)
    stride2 = (4, 1)
    dtype2 = torch.float32
    pin_memory2 = True
    requires_grad2 = True
    
    input_dict2 = {
        "size": input2,
        "stride": stride2,
        "dtype": dtype2,
        "pin_memory": pin_memory2,
        "requires_grad": requires_grad2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = (2, 3, 4)
    stride3 = (1, 2, 4)
    dtype3 = torch.complex64
    pin_memory3 = False
    requires_grad3 = False
    
    input_dict3 = {
        "size": input3,
        "stride": stride3,
        "dtype": dtype3,
        "pin_memory": pin_memory3,
        "requires_grad": requires_grad3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    return list_of_inputs

generated_inputs["torch.empty_strided"] = empty_strided_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.empty_strided' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.empty_strided'.")


check_valid('torch.empty_strided', generated_inputs['torch.empty_strided'], lib="torch", suffix=0)
