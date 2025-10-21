
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def as_strided_inputs():
    list_of_inputs = []
    
    input1 = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    size1 = (2, 3, 4)
    stride1 = (4, 8, 1)
    storage_offset1 = 0
    
    input_dict1 = {
        "input": input1,
        "size": size1,
        "stride": stride1,
        "storage_offset": storage_offset1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.arange(12, dtype=np.int64).reshape(3, 4)
    size2 = (3, 4)
    stride2 = (4, 1)
    storage_offset2 = 0
    
    input_dict2 = {
        "input": input2,
        "size": size2,
        "stride": stride2,
        "storage_offset": storage_offset2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    size3 = (2, 2, 2)
    stride3 = (2, 4, 1)
    storage_offset3 = 0
    
    input_dict3 = {
        "input": input3,
        "size": size3,
        "stride": stride3,
        "storage_offset": storage_offset3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.arange(6, dtype=np.int32).reshape(1, 6)
    size4 = (1, 6)
    stride4 = (1, 1)
    storage_offset4 = 0
    
    input_dict4 = {
        "input": input4,
        "size": size4,
        "stride": stride4,
        "storage_offset": storage_offset4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.arange(10, dtype=np.float32).reshape(5, 2)
    size5 = (5, 2)
    stride5 = (2, 1)
    storage_offset5 = 0
    
    input_dict5 = {
        "input": input5,
        "size": size5,
        "stride": stride5,
        "storage_offset": storage_offset5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.arange(3, dtype=np.int64).reshape(1, 1, 3)
    size6 = (1, 1, 3)
    stride6 = (3, 1, 1)
    storage_offset6 = 0
    
    input_dict6 = {
        "input": input6,
        "size": size6,
        "stride": stride6,
        "storage_offset": storage_offset6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.as_strided"] = as_strided_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.as_strided' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_strided'.")


check_valid('torch.as_strided', generated_inputs['torch.as_strided'], lib="torch", suffix=0)
