
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def narrow_copy_inputs():
    list_of_inputs = []

    input1 = np.random.rand(5, 4, 3).astype(np.float32)
    dim1 = 1
    start1 = 0
    length1 = 3
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "start": start1,
        "length": length1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randint(0, 10, (3,)).astype(np.int64)
    dim2 = 0
    start2 = 1
    length2 = 2
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "start": start2,
        "length": length2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(2, 2).astype(np.float64)
    dim3 = 0
    start3 = 0
    length3 = 1
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "start": start3,
        "length": length3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randint(0, 5, (4, 1, 2)).astype(np.int32)
    dim4 = 2
    start4 = 0
    length4 = 1
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "start": start4,
        "length": length4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(6, 5, 4, 3).astype(np.float16)
    dim5 = 3
    start5 = 0
    length5 = 2
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "start": start5,
        "length": length5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.randint(0, 10, (2, 3, 4)).astype(np.int64)
    dim6 = 1
    start6 = 0
    length6 = 2
    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "start": start6,
        "length": length6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(3, 3).astype(np.float32)
    dim7 = 0
    start7 = 0
    length7 = 2
    input_dict7 = {
        "input": input7,
        "dim": dim7,
        "start": start7,
        "length": length7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs["torch.narrow_copy"] = narrow_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.narrow_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.narrow_copy'.")


check_valid('torch.narrow_copy', generated_inputs['torch.narrow_copy'], lib="torch", suffix=0)
