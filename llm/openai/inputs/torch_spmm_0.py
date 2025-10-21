
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def spmm_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    mat2_1 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    list_of_inputs.append({"input": input1, "mat2": mat2_1})
    
    input2 = np.array([[1, 0], [0, 1]], dtype=np.float64)
    mat2_2 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    list_of_inputs.append({"input": input2, "mat2": mat2_2})
    
    input3 = np.array([[-1, 0], [0, -1]], dtype=np.float32)
    mat2_3 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    list_of_inputs.append({"input": input3, "mat2": mat2_3})

    input4 = np.array([[1, 0], [0, 1]], dtype=np.float32)
    mat2_4 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    list_of_inputs.append({"input": input4, "mat2": mat2_4})

    input5 = np.array([[0.5, 0.2], [0.1, 0.8]], dtype=np.float32)
    mat2_5 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    list_of_inputs.append({"input": input5, "mat2": mat2_5})

    return list_of_inputs

generated_inputs["torch.spmm"] = spmm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.spmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.spmm'.")


check_valid('torch.spmm', generated_inputs['torch.spmm'], lib="torch", suffix=0)
