
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trunc_inputs():
    list_of_inputs = []

    # Input 1: Simple positive float
    list_of_inputs.append({"x": 3.14})

    # Input 2: Simple negative float
    list_of_inputs.append({"x": -3.14})

    # Input 3: Float close to zero (positive)
    list_of_inputs.append({"x": 0.1234})

    # Input 4: Float close to zero (negative)
    list_of_inputs.append({"x": -0.9999})

    # Input 5: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 6: Negative zero
    list_of_inputs.append({"x": -0.0})

    # Input 7: Large positive float
    list_of_inputs.append({"x": 1234567.89})

    # Input 8: Large negative float
    list_of_inputs.append({"x": -9876543.21})

    # Input 9: Infinity
    list_of_inputs.append({"x": float('inf')})

    # Input 10: Negative infinity
    list_of_inputs.append({"x": float('-inf')})

    # Input 11: NaN (Not a Number)
    list_of_inputs.append({"x": float('nan')})

    return list_of_inputs

generated_inputs["jax.numpy.trunc_2"] = trunc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trunc_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trunc_2'.")


check_valid('jax.numpy.trunc', generated_inputs['jax.numpy.trunc_2'], lib="jax", suffix=2)
