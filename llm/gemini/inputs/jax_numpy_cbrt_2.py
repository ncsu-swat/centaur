
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cbrt_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive integer represented as float
    input_dict = {"x": 8.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer represented as float
    input_dict = {"x": -27.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: One
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative one
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Fractional positive float
    input_dict = {"x": 0.125}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Fractional negative float
    input_dict = {"x": -0.008}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large positive float
    input_dict = {"x": 1e12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small positive float
    input_dict = {"x": 1e-15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative infinity
    input_dict = {"x": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Arbitrary decimal float
    input_dict = {"x": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
