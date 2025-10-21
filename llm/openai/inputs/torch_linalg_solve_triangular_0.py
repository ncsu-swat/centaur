
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def solve_triangular_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=np.float64)
    input2 = np.array([7, 8, 9], dtype=np.float64)
    upper1 = True
    unitriangular1 = False
    
    input_dict1 = {
        "a": input1,
        "b": input2,
        "upper": upper1,
        "unitriangular": unitriangular1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input3 = np.array([[10, 0, 0], [11, 12, 0], [13, 14, 15]], dtype=np.float32)
    input4 = np.array([16, 17, 18], dtype=np.float32)
    upper3 = False
    unitriangular3 = False
    
    input_dict3 = {
        "a": input3,
        "b": input4,
        "upper": upper3,
        "unitriangular": unitriangular3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.linalg.solve_triangular"] = solve_triangular_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.solve_triangular' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.solve_triangular'.")


check_valid('torch.linalg.solve_triangular', generated_inputs['torch.linalg.solve_triangular'], lib="torch", suffix=0)
