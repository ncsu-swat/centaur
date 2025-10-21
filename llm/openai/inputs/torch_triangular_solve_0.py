
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def triangular_solve_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
    A1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    upper1 = True
    transpose1 = False
    unitriangular1 = True
    out1 = (torch.empty((2, 2), dtype=torch.float64).numpy(), torch.empty((2, 2), dtype=torch.float64).numpy())

    input_dict1 = {
        "input": input1,
        "A": A1,
        "upper": upper1,
        "transpose": transpose1,
        "unitriangular": unitriangular1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]], dtype=np.float32)
    A2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    upper2 = False
    transpose2 = True
    unitriangular2 = False
    out2 = (torch.empty((3, 3), dtype=torch.float32).numpy(), torch.empty((3, 3), dtype=torch.float32).numpy())

    input_dict2 = {
        "input": input2,
        "A": A2,
        "upper": upper2,
        "transpose": transpose2,
        "unitriangular": unitriangular2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.triangular_solve"] = triangular_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.triangular_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.triangular_solve'.")


check_valid('torch.triangular_solve', generated_inputs['torch.triangular_solve'], lib="torch", suffix=0)
