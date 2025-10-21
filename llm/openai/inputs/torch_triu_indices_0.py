
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def triu_indices_inputs():
    list_of_inputs = []
    
    input1 = np.array(5, dtype=np.int32)
    input2 = np.array(5, dtype=np.int32)
    input3 = np.array(0, dtype=np.int32)
    input4 = np.dtype(np.int64)
    
    input_dict1 = {
        "row": input1,
        "col": input2,
        "offset": input3,
        "dtype": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input1 = np.array(10, dtype=np.int64)
    input2 = np.array(10, dtype=np.int64)
    input3 = np.array(1, dtype=np.int64)
    input4 = np.dtype(np.int32)
    
    input_dict2 = {
        "row": input1,
        "col": input2,
        "offset": input3,
        "dtype": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input1 = np.array(3, dtype=np.int32)
    input2 = np.array(3, dtype=np.int32)
    input3 = np.array(-1, dtype=np.int32)
    input4 = np.dtype(np.int64)
    
    input_dict3 = {
        "row": input1,
        "col": input2,
        "offset": input3,
        "dtype": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input1 = np.array(4, dtype=np.int32)
    input2 = np.array(4, dtype=np.int32)
    input3 = np.array(2, dtype=np.int32)
    input4 = np.dtype(np.int32)
    
    input_dict4 = {
        "row": input1,
        "col": input2,
        "offset": input3,
        "dtype": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input1 = np.array(6, dtype=np.int64)
    input2 = np.array(6, dtype=np.int64)
    input3 = np.array(-2, dtype=np.int64)
    input4 = np.dtype(np.int64)
    
    input_dict5 = {
        "row": input1,
        "col": input2,
        "offset": input3,
        "dtype": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input1 = np.array(8, dtype=np.int32)
    input2 = np.array(8, dtype=np.int32)
    input3 = np.array(1, dtype=np.int32)
    input4 = np.dtype(np.int32)
    
    input_dict6 = {
        "row": input1,
        "col": input2,
        "offset": input3,
        "dtype": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input1 = np.array(9, dtype=np.int64)
    input2 = np.array(9, dtype=np.int64)
    input3 = np.array(-1, dtype=np.int64)
    input4 = np.dtype(np.int64)
    
    input_dict7 = {
        "row": input1,
        "col": input2,
        "offset": input3,
        "dtype": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.triu_indices"] = triu_indices_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.triu_indices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.triu_indices'.")


check_valid('torch.triu_indices', generated_inputs['torch.triu_indices'], lib="torch", suffix=0)
