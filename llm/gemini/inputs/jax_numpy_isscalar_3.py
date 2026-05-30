
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isscalar_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    list_of_inputs.append({"element": 1.0})

    # Input 2: Zero
    list_of_inputs.append({"element": 0.0})

    # Input 3: Standard negative float
    list_of_inputs.append({"element": -3.14159})

    # Input 4: Large positive float
    list_of_inputs.append({"element": 1e10})

    # Input 5: Small positive float
    list_of_inputs.append({"element": 1e-10})

    # Input 6: Positive infinity
    list_of_inputs.append({"element": float('inf')})

    # Input 7: Negative infinity
    list_of_inputs.append({"element": float('-inf')})

    # Input 8: NaN (Not a Number)
    list_of_inputs.append({"element": float('nan')})

    # Input 9: Max float64 representation
    list_of_inputs.append({"element": float(np.finfo(np.float64).max)})

    # Input 10: Min float64 representation (closest to zero)
    list_of_inputs.append({"element": float(np.finfo(np.float64).tiny)})

    # Input 11: Machine epsilon for float32
    list_of_inputs.append({"element": float(np.finfo(np.float32).eps)})

    return list_of_inputs

generated_inputs["jax.numpy.isscalar_3"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_3'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_3'], lib="jax", suffix=3)
