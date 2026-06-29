
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_max_inputs():
    list_of_inputs = []

    # Input 1: positive floats
    list_of_inputs.append({"x": 1.5, "y": 2.5})

    # Input 2: negative floats
    list_of_inputs.append({"x": -3.0, "y": -1.2})

    # Input 3: mix of positive and negative
    list_of_inputs.append({"x": -5.5, "y": 4.5})

    # Input 4: zeroes
    list_of_inputs.append({"x": 0.0, "y": 0.0})

    # Input 5: signed zeroes
    list_of_inputs.append({"x": -0.0, "y": 0.0})

    # Input 6: large floats
    list_of_inputs.append({"x": 1e10, "y": 1e11})

    # Input 7: small fractional floats
    list_of_inputs.append({"x": 1e-5, "y": 2e-5})

    # Input 8: infinity values
    list_of_inputs.append({"x": float('inf'), "y": float('-inf')})

    # Input 9: close floats
    list_of_inputs.append({"x": 1.0000001, "y": 1.0})

    # Input 10: large negative and positive floats
    list_of_inputs.append({"x": -99999.9, "y": 99999.9})

    return list_of_inputs

generated_inputs["jax.lax.max_3"] = jax_lax_max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.max_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.max_3'.")


check_valid('jax.lax.max', generated_inputs['jax.lax.max_3'], lib="jax", suffix=3)
