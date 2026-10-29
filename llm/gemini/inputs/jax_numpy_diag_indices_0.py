
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def diag_indices_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D diagonal indices
    list_of_inputs.append({"n": 3, "ndim": 2})

    # Input 2: 3D diagonal indices
    list_of_inputs.append({"n": 4, "ndim": 3})

    # Input 3: 1D case
    list_of_inputs.append({"n": 5, "ndim": 1})

    # Input 4: 4D case with small size
    list_of_inputs.append({"n": 2, "ndim": 4})

    # Input 5: n = 0 (empty diagonal)
    list_of_inputs.append({"n": 0, "ndim": 2})

    # Input 6: n = 1 (single element)
    list_of_inputs.append({"n": 1, "ndim": 2})

    # Input 7: ndim = 0 (0-dimensional indexing tuple)
    list_of_inputs.append({"n": 3, "ndim": 0})

    # Input 8: Larger n with 2D
    list_of_inputs.append({"n": 100, "ndim": 2})

    # Input 9: High-dimensional array
    list_of_inputs.append({"n": 3, "ndim": 6})

    # Input 10: Larger n and 3D
    list_of_inputs.append({"n": 16, "ndim": 3})

    # Input 11: Using np.int32
    list_of_inputs.append({"n": int(np.int32(8)), "ndim": int(np.int32(2))})

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
