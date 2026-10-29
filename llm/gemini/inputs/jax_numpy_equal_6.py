
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def equal_inputs():
    list_of_inputs = []

    # Input 1: Identical positive floats
    list_of_inputs.append({"x": 1.0, "y": 1.0})

    # Input 2: Different positive floats
    list_of_inputs.append({"x": 2.5, "y": 3.75})

    # Input 3: Positive zero and negative zero
    list_of_inputs.append({"x": 0.0, "y": -0.0})

    # Input 4: Identical negative floats
    list_of_inputs.append({"x": -4.2, "y": -4.2})

    # Input 5: Positive and negative floats of same magnitude
    list_of_inputs.append({"x": 10.0, "y": -10.0})

    # Input 6: Large float values equal
    list_of_inputs.append({"x": 1e20, "y": 1e20})

    # Input 7: Small float values unequal
    list_of_inputs.append({"x": 1e-12, "y": 2e-12})

    # Input 8: Infinities of same sign
    list_of_inputs.append({"x": float("inf"), "y": float("inf")})

    # Input 9: Infinities of opposite signs
    list_of_inputs.append({"x": float("-inf"), "y": float("inf")})

    # Input 10: NaN values
    list_of_inputs.append({"x": float("nan"), "y": float("nan")})

    # Input 11: Subnormal-like small floats
    list_of_inputs.append({"x": 5e-324, "y": 5e-324})

    # Input 12: High precision floats
    list_of_inputs.append({"x": 3.141592653589793, "y": 3.141592653589793})

    return list_of_inputs

generated_inputs["jax.numpy.equal_6"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_6'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_6'], lib="jax", suffix=6)
