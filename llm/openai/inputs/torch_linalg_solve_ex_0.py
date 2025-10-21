
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def solve_ex_inputs():
    list_of_inputs = []

    A1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    B1 = np.array([5.0, 6.0])
    left1 = True
    check_errors1 = True

    input_dict1 = {
        "A": A1,
        "B": B1,
        "left": left1,
        "check_errors": check_errors1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.linalg.solve_ex"] = solve_ex_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.solve_ex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.solve_ex'.")


check_valid('torch.linalg.solve_ex', generated_inputs['torch.linalg.solve_ex'], lib="torch", suffix=0)
