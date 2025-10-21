
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def autocast_increment_nesting_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({
        "torch.autocast_increment_nesting": input1
    })

    input2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({
        "torch.autocast_increment_nesting": input2
    })

    input3 = np.array([[-1, -2], [-3, -4]], dtype=np.int64)
    list_of_inputs.append({
        "torch.autocast_increment_nesting": input3
    })

    input4 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    list_of_inputs.append({
        "torch.autocast_increment_nesting": input4
    })

    input5 = np.array([], dtype=np.int32)
    list_of_inputs.append({
        "torch.autocast_increment_nesting": input5
    })

    input6 = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    list_of_inputs.append({
        "torch.autocast_increment_nesting": input6
    })

    input7 = np.array([[1, 0], [0, 1]], dtype=np.bool_)
    list_of_inputs.append({
        "torch.autocast_increment_nesting": input7
    })

    input8 = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    list_of_inputs.append({
        "torch.autocast_increment_nesting": input8
    })

    input9 = np.array([1, 2, 3], dtype=np.uint32)
    list_of_inputs.append({
        "torch.autocast_increment_nesting": input9
    })

    input10 = np.array([1000, 2000, 3000], dtype=np.int64)
    list_of_inputs.append({
        "torch.autocast_increment_nesting": input10
    })

    return list_of_inputs

generated_inputs["torch.autocast_increment_nesting"] = autocast_increment_nesting_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.autocast_increment_nesting' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.autocast_increment_nesting'.")


check_valid('torch.autocast_increment_nesting', generated_inputs['torch.autocast_increment_nesting'], lib="torch", suffix=0)
