
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def nansum_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    dim1 = 0
    keepdim1 = False
    dtype1 = torch.float64
    
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    dim2 = 1
    keepdim2 = True
    dtype2 = torch.float32
    
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, 2.0], [3.0, -np.nan]], dtype=np.float32)
    dim3 = None
    keepdim3 = False
    dtype3 = torch.float32
    
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "keepdim": keepdim3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([[[1.0, np.nan], [2.0, 3.0]], [[4.0, 5.0], [np.nan, 7.0]]], dtype=np.float32)
    dim4 = (0, 1)
    keepdim4 = False
    dtype4 = torch.float64
    
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "keepdim": keepdim4,
        "dtype": dtype4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dim5 = 0
    keepdim5 = False
    dtype5 = torch.float32
    
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "keepdim": keepdim5,
        "dtype": dtype5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    dim6 = None
    keepdim6 = True
    dtype6 = torch.float32
    
    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "keepdim": keepdim6,
        "dtype": dtype6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    dim7 = 0
    keepdim7 = False
    dtype7 = torch.float32
    
    input_dict7 = {
        "input": input7,
        "dim": dim7,
        "keepdim": keepdim7,
        "dtype": dtype7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    dim8 = 1
    keepdim8 = False
    dtype8 = torch.float32
    
    input_dict8 = {
        "input": input8,
        "dim": dim8,
        "keepdim": keepdim8,
        "dtype": dtype8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.nansum_2"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nansum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_2'.")


check_valid('torch.nansum', generated_inputs['torch.nansum_2'], lib="torch", suffix=2)
