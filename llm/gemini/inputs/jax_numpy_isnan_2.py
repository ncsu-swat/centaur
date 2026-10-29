
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isnan_inputs():
    list_of_inputs = []

    # Input 1: NaN float
    list_of_inputs.append({"x": float("nan")})

    # Input 2: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 3: Negative zero
    list_of_inputs.append({"x": -0.0})

    # Input 4: Positive infinity
    list_of_inputs.append({"x": float("inf")})

    # Input 5: Negative infinity
    list_of_inputs.append({"x": float("-inf")})

    # Input 6: Standard positive float
    list_of_inputs.append({"x": 1.0})

    # Input 7: Standard negative float
    list_of_inputs.append({"x": -42.5})

    # Input 8: Very small positive float
    list_of_inputs.append({"x": 1e-30})

    # Input 9: Very large positive float
    list_of_inputs.append({"x": 1e30})

    # Input 10: Mathematical constant pi approximation
    list_of_inputs.append({"x": 3.141592653589793})

    # Input 11: Negative fraction
    list_of_inputs.append({"x": -0.125})

    # Input 12: Subnormal-like small float
    list_of_inputs.append({"x": 5e-324})

    return list_of_inputs


generated_inputs["jax.numpy.isnan_2"] = isnan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isnan_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isnan_2'.")


check_valid('jax.numpy.isnan', generated_inputs['jax.numpy.isnan_2'], lib="jax", suffix=2)
