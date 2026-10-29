
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fabs_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 2: Negative zero
    list_of_inputs.append({"x": -0.0})

    # Input 3: Negative float
    list_of_inputs.append({"x": -1.5})

    # Input 4: Positive float
    list_of_inputs.append({"x": 3.141592653589793})

    # Input 5: Large negative float
    list_of_inputs.append({"x": -1e10})

    # Input 6: Small positive float
    list_of_inputs.append({"x": 1e-10})

    # Input 7: Negative integer-valued float
    list_of_inputs.append({"x": -42.0})

    # Input 8: Positive infinity
    list_of_inputs.append({"x": float("inf")})

    # Input 9: Negative infinity
    list_of_inputs.append({"x": float("-inf")})

    # Input 10: NaN
    list_of_inputs.append({"x": float("nan")})

    # Input 11: Negative fractional float
    list_of_inputs.append({"x": -2.718281828459045})

    # Input 12: Positive large float
    list_of_inputs.append({"x": 65536.125})

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.numpy.fabs_3"] = fabs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fabs_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fabs_3'.")


check_valid('jax.numpy.fabs', generated_inputs['jax.numpy.fabs_3'], lib="jax", suffix=3)
