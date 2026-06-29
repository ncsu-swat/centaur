
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_lax_eq_inputs():
    list_of_inputs = []

    # Input 1: Equal positive floats
    list_of_inputs.append({"x": 1.0, "y": 1.0})

    # Input 2: Unequal positive floats
    list_of_inputs.append({"x": 1.0, "y": 2.5})

    # Input 3: Equal negative floats
    list_of_inputs.append({"x": -3.14, "y": -3.14})

    # Input 4: Unequal negative floats
    list_of_inputs.append({"x": -0.5, "y": -1.5})

    # Input 5: Zeroes
    list_of_inputs.append({"x": 0.0, "y": 0.0})

    # Input 6: Zero and positive float
    list_of_inputs.append({"x": 0.0, "y": 0.0001})

    # Input 7: Large values
    list_of_inputs.append({"x": 1e9, "y": 1e9})

    # Input 8: Very small values
    list_of_inputs.append({"x": 1e-9, "y": -1e-9})

    # Input 9: Infinity
    list_of_inputs.append({"x": float("inf"), "y": float("inf")})

    # Input 10: Negative infinity and positive infinity
    list_of_inputs.append({"x": float("-inf"), "y": float("inf")})

    # Input 11: NaN
    list_of_inputs.append({"x": float("nan"), "y": float("nan")})

    return list_of_inputs


generated_inputs["jax.lax.eq_3"] = jax_lax_eq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.eq_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.eq_3'.")


check_valid('jax.lax.eq', generated_inputs['jax.lax.eq_3'], lib="jax", suffix=3)
