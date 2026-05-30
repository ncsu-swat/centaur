
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def remainder_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats
    list_of_inputs.append({"x1": 5.0, "x2": 2.0})

    # Input 2: Negative dividend, positive divisor
    list_of_inputs.append({"x1": -5.0, "x2": 2.0})

    # Input 3: Positive dividend, negative divisor
    list_of_inputs.append({"x1": 5.0, "x2": -2.0})

    # Input 4: Both negative floats
    list_of_inputs.append({"x1": -5.0, "x2": -2.0})

    # Input 5: Zero dividend
    list_of_inputs.append({"x1": 0.0, "x2": 3.5})

    # Input 6: Fractional values
    list_of_inputs.append({"x1": 7.3, "x2": 2.1})

    # Input 7: Large values
    list_of_inputs.append({"x1": 123456.78, "x2": 123.45})

    # Input 8: Very small values
    list_of_inputs.append({"x1": 0.0001, "x2": 0.00003})

    # Input 9: Divisor larger than dividend
    list_of_inputs.append({"x1": 1.5, "x2": 10.0})

    # Input 10: Negative divisor larger in magnitude
    list_of_inputs.append({"x1": 2.5, "x2": -10.0})

    return list_of_inputs

generated_inputs["jax.numpy.remainder_3"] = remainder_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.remainder_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.remainder_3'.")


check_valid('jax.numpy.remainder', generated_inputs['jax.numpy.remainder_3'], lib="jax", suffix=3)
