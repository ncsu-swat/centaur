
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def select_copy_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    dim1 = 0
    index1 = 2
    out1 = np.array([], dtype=np.int64)

    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "index": index1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    dim2 = 1
    index2 = 0
    out2 = np.array([], dtype=np.int64)

    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "index": index2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    dim3 = 0
    index3 = 4
    out3 = np.array([], dtype=np.int64)

    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "index": index3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    dim4 = 1
    index4 = 1
    out4 = np.array([], dtype=np.int64)

    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "index": index4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    dim5 = 2
    index5 = 1
    out5 = np.array([], dtype=np.int64)

    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "index": index5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    dim6 = 0
    index6 = -1
    out6 = np.array([], dtype=np.int64)

    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "index": index6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.select_copy"] = select_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.select_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.select_copy'.")


check_valid('torch.select_copy', generated_inputs['torch.select_copy'], lib="torch", suffix=0)
