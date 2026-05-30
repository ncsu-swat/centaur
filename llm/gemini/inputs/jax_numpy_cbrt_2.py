
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cbrt_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    list_of_inputs.append({"x": 8.0})

    # Input 2: Standard negative float
    list_of_inputs.append({"x": -27.0})

    # Input 3: Zero float
    list_of_inputs.append({"x": 0.0})

    # Input 4: Negative zero float
    list_of_inputs.append({"x": -0.0})

    # Input 5: Small positive float
    list_of_inputs.append({"x": 1e-6})

    # Input 6: Small negative float
    list_of_inputs.append({"x": -1e-6})

    # Input 7: Large positive float
    list_of_inputs.append({"x": 1e15})

    # Input 8: Large negative float
    list_of_inputs.append({"x": -1e15})

    # Input 9: Positive Infinity
    list_of_inputs.append({"x": float('inf')})

    # Input 10: Negative Infinity
    list_of_inputs.append({"x": float('-inf')})

    # Input 11: NaN (Not a Number) float
    list_of_inputs.append({"x": float('nan')})

    # Input 12: Decimal float
    list_of_inputs.append({"x": 3.1415926535})

    return list_of_inputs

generated_inputs["jax.numpy.cbrt_2"] = cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cbrt_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cbrt_2'.")


check_valid('jax.numpy.cbrt', generated_inputs['jax.numpy.cbrt_2'], lib="jax", suffix=2)
