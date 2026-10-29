
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atleast_1d_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"arys": 0.0})

    # Input 2: Positive integer as float
    list_of_inputs.append({"arys": 1.0})

    # Input 3: Negative float
    list_of_inputs.append({"arys": -1.5})

    # Input 4: Large positive float
    list_of_inputs.append({"arys": 1e6})

    # Input 5: Small positive float
    list_of_inputs.append({"arys": 1e-7})

    # Input 6: Large negative float
    list_of_inputs.append({"arys": -1e6})

    # Input 7: Pi approximation
    list_of_inputs.append({"arys": 3.141592653589793})

    # Input 8: Positive infinity
    list_of_inputs.append({"arys": float("inf")})

    # Input 9: Negative infinity
    list_of_inputs.append({"arys": float("-inf")})

    # Input 10: NaN
    list_of_inputs.append({"arys": float("nan")})

    # Input 11: Negative zero
    list_of_inputs.append({"arys": -0.0})

    # Input 12: Arbitrary float
    list_of_inputs.append({"arys": 42.42})

    return [copy.deepcopy(x) for x in list_of_inputs]


generated_inputs["jax.numpy.atleast_1d_3"] = atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_1d_3'.")


check_valid('jax.numpy.atleast_1d', generated_inputs['jax.numpy.atleast_1d_3'], lib="jax", suffix=3)
