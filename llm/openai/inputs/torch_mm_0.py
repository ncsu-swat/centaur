
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def mm_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(2, 3).numpy()
    mat2_1 = torch.randn(3, 3).numpy()
    out1 = np.zeros((2, 3)).astype(np.float32)

    input_dict1 = {
        "input": input1,
        "mat2": mat2_1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(4, 2).numpy()
    mat2_2 = torch.randn(2, 5).numpy()
    out2 = np.zeros((4, 5)).astype(np.float32)

    input_dict2 = {
        "input": input2,
        "mat2": mat2_2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(low=-10, high=10, size=(3, 4)).numpy()
    mat2_3 = torch.randint(low=-5, high=5, size=(4, 2)).numpy()
    out3 = np.zeros((3, 2)).astype(np.int64)

    input_dict3 = {
        "input": input3,
        "mat2": mat2_3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.mm"] = mm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.mm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mm'.")


check_valid('torch.mm', generated_inputs['torch.mm'], lib="torch", suffix=0)
