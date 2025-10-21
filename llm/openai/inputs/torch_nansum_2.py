
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_nansum_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float64)
    input_dict2 = {"input": input2, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[1.0, np.nan], [np.nan, 3.0]], dtype=np.float32)
    input_dict3 = {"input": input3, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float64)
    input_dict4 = {"input": input4, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[[1.0, 2.0], [3.0, np.nan]], [[np.nan, 5.0], [6.0, 7.0]]], dtype=np.float32)
    input_dict5 = {"input": input5, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1.0, 2.0], [3.0, np.nan]], [[np.nan, 5.0], [6.0, 7.0]]], dtype=np.float32)
    input_dict6 = {"input": input6, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.nansum_2"] = torch_nansum_inputs()

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
