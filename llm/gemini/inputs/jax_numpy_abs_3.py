
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import math

def abs_inputs():
    list_of_inputs = []

    # Input 1: positive float
    list_of_inputs.append({"x": 1.0})

    # Input 2: negative float
    list_of_inputs.append({"x": -1.0})

    # Input 3: zero
    list_of_inputs.append({"x": 0.0})

    # Input 4: negative zero
    list_of_inputs.append({"x": -0.0})

    # Input 5: positive large float
    list_of_inputs.append({"x": 1e20})

    # Input 6: negative large float
    list_of_inputs.append({"x": -1e20})

    # Input 7: positive small float (subnormal range)
    list_of_inputs.append({"x": 1e-30})

    # Input 8: negative fractional float
    list_of_inputs.append({"x": -3.141592653589793})

    # Input 9: positive infinity
    list_of_inputs.append({"x": float("inf")})

    # Input 10: negative infinity
    list_of_inputs.append({"x": float("-inf")})

    # Input 11: NaN
    list_of_inputs.append({"x": float("nan")})

    # Input 12: arbitrary negative float
    list_of_inputs.append({"x": -42.42})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.numpy.abs_3"] = abs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.abs_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.abs_3'.")


check_valid('jax.numpy.abs', generated_inputs['jax.numpy.abs_3'], lib="jax", suffix=3)
