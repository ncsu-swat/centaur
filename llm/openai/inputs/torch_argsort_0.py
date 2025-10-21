
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def argsort_inputs():
    list_of_inputs = []

    input1 = np.random.rand(5).astype(np.float32)
    dim1 = 0
    descending1 = False
    stable1 = False
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "descending": descending1,
        "stable": stable1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(2, 3).astype(np.float64)
    dim2 = 1
    descending2 = True
    stable2 = True
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "descending": descending2,
        "stable": stable2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randint(-10, 10, size=(4, 4, 2)).astype(np.int32)
    dim3 = 2
    descending3 = False
    stable3 = False
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "descending": descending3,
        "stable": stable3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(3).astype(np.float16)
    dim4 = 0
    descending4 = True
    stable4 = True
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "descending": descending4,
        "stable": stable4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(6, 1).astype(np.float32)
    dim5 = 0
    descending5 = False
    stable5 = False
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "descending": descending5,
        "stable": stable5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(2, 2, 2).astype(np.float64)
    dim6 = 1
    descending6 = True
    stable6 = True
    input_dict6 = {
        "input": input6,
        "dim": dim6,
        "descending": descending6,
        "stable": stable6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.randint(0, 5, size=(3, 3)).astype(np.int64)
    dim7 = 1
    descending7 = False
    stable7 = False
    input_dict7 = {
        "input": input7,
        "dim": dim7,
        "descending": descending7,
        "stable": stable7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(7, 2, 3).astype(np.float32)
    dim8 = 2
    descending8 = True
    stable8 = True
    input_dict8 = {
        "input": input8,
        "dim": dim8,
        "descending": descending8,
        "stable": stable8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(4).astype(np.float16)
    dim9 = 0
    descending9 = False
    stable9 = True
    input_dict9 = {
        "input": input9,
        "dim": dim9,
        "descending": descending9,
        "stable": stable9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.randint(-5, 5, size=(2, 4)).astype(np.int32)
    dim10 = 1
    descending10 = True
    stable10 = False
    input_dict10 = {
        "input": input10,
        "dim": dim10,
        "descending": descending10,
        "stable": stable10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.argsort"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.argsort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argsort'.")


check_valid('torch.argsort', generated_inputs['torch.argsort'], lib="torch", suffix=0)
