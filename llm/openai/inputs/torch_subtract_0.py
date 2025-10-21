
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def subtract_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    tensor1 = np.array([0.1, 0.2, 0.3])
    value1 = 1.0
    out1 = np.array([])

    input_dict1 = {
        "input": input1,
        "other": tensor1,
        "alpha": value1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.ones((2, 3))
    tensor2 = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
    value2 = 0.5
    out2 = np.zeros((2, 3))

    input_dict2 = {
        "input": input2,
        "other": tensor2,
        "alpha": value2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    tensor3 = np.array([[0.5, -0.5], [-1.5, 2.5]])
    value3 = 2.0
    out3 = np.zeros((2, 2))

    input_dict3 = {
        "input": input3,
        "other": tensor3,
        "alpha": value3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    
    input4 = np.array([1, 2, 3], dtype=np.int32)
    tensor4 = np.array([1, 2, 3], dtype=np.int32)
    value4 = 1
    out4 = np.array([])

    input_dict4 = {
        "input": input4,
        "other": tensor4,
        "alpha": value4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.subtract"] = subtract_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.subtract' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.subtract'.")


check_valid('torch.subtract', generated_inputs['torch.subtract'], lib="torch", suffix=0)
