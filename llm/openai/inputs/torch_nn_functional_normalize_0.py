
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def normalize_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    p1 = 2.0
    dim1 = 0
    eps1 = 1e-6
    out1 = np.array([], dtype=np.float32)

    input_dict1 = {
        "input": input1,
        "p": p1,
        "dim": dim1,
        "eps": eps1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    p2 = 1.0
    dim2 = 0
    eps2 = 1e-8
    out2 = np.array([], dtype=np.float32)

    input_dict2 = {
        "input": input2,
        "p": p2,
        "dim": dim2,
        "eps": eps2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    p3 = 2.0
    dim3 = 0
    eps3 = 1e-5
    out3 = np.array([], dtype=np.float32)

    input_dict3 = {
        "input": input3,
        "p": p3,
        "dim": dim3,
        "eps": eps3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.nn.functional.normalize"] = normalize_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.normalize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.normalize'.")


check_valid('torch.nn.functional.normalize', generated_inputs['torch.nn.functional.normalize'], lib="torch", suffix=0)
