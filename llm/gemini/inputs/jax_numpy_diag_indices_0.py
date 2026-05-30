
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def diag_indices_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D case, small size
    list_of_inputs.append({"n": int(3), "ndim": int(2)})

    # Input 2: 3D case, small size
    list_of_inputs.append({"n": int(4), "ndim": int(3)})

    # Input 3: Minimal size and dimension
    list_of_inputs.append({"n": int(1), "ndim": int(1)})

    # Input 4: Boundary case with size 0
    list_of_inputs.append({"n": int(0), "ndim": int(2)})

    # Input 5: High dimensions, small size
    list_of_inputs.append({"n": int(2), "ndim": int(5)})

    # Input 6: Large size, 2D
    list_of_inputs.append({"n": int(100), "ndim": int(2)})

    # Input 7: Moderate size and dimensions
    list_of_inputs.append({"n": int(5), "ndim": int(4)})

    # Input 8: 1D array diagonal indices
    list_of_inputs.append({"n": int(8), "ndim": int(1)})

    # Input 9: Symmetrical dimensions and size
    list_of_inputs.append({"n": int(6), "ndim": int(6)})

    # Input 10: Moderate size, 3D
    list_of_inputs.append({"n": int(10), "ndim": int(3)})

    # Input 11: High size, 2D
    list_of_inputs.append({"n": int(50), "ndim": int(2)})

    return list_of_inputs

generated_inputs["jax.numpy.diag_indices"] = diag_indices_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.diag_indices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.diag_indices'.")


check_valid('jax.numpy.diag_indices', generated_inputs['jax.numpy.diag_indices'], lib="jax", suffix=0)
