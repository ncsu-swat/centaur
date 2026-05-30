
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def conjugate_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    list_of_inputs.append({"x": 3.14})

    # Input 2: Standard negative float
    list_of_inputs.append({"x": -2.718})

    # Input 3: Zero float
    list_of_inputs.append({"x": 0.0})

    # Input 4: Negative zero float
    list_of_inputs.append({"x": -0.0})

    # Input 5: Large positive float
    list_of_inputs.append({"x": 1.23e10})

    # Input 6: Small positive float
    list_of_inputs.append({"x": 4.56e-10})

    # Input 7: Positive infinity
    list_of_inputs.append({"x": float('inf')})

    # Input 8: Negative infinity
    list_of_inputs.append({"x": float('-inf')})

    # Input 9: Not a Number (NaN)
    list_of_inputs.append({"x": float('nan')})

    # Input 10: Arbitrary float value
    list_of_inputs.append({"x": 42.0})

    return list_of_inputs

generated_inputs["jax.numpy.conjugate_3"] = conjugate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.conjugate_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.conjugate_3'.")


check_valid('jax.numpy.conjugate', generated_inputs['jax.numpy.conjugate_3'], lib="jax", suffix=3)
