
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reciprocal_inputs():
    list_of_inputs = []

    # Input 1: Basic positive float
    input_dict = {"x": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Unit float
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float
    input_dict = {"x": -4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive fraction
    input_dict = {"x": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative fraction
    input_dict = {"x": -0.125}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero (results in inf)
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative zero (results in -inf)
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small positive float
    input_dict = {"x": 1e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very large positive float
    input_dict = {"x": 1e8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Transcendental number float (pi)
    input_dict = {"x": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative infinity
    input_dict = {"x": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.reciprocal_2"] = reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.reciprocal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.reciprocal_2'.")


check_valid('jax.numpy.reciprocal', generated_inputs['jax.numpy.reciprocal_2'], lib="jax", suffix=2)
