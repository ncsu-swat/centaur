
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: x1 is zero, x2 is positive integer
    input_dict = {"x1": 0.0, "x2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: x1 is negative, x2 is zero
    input_dict = {"x1": -2.5, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: x1 is positive, x2 is negative integer
    input_dict = {"x1": 3.14159, "x2": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: x1 is zero, x2 is zero
    input_dict = {"x1": 0.0, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x1 is negative zero, x2 is positive integer
    input_dict = {"x1": -0.0, "x2": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x1 is a large positive float, x2 is positive integer
    input_dict = {"x1": 1e6, "x2": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x1 is a small negative float, x2 is a large integer
    input_dict = {"x1": -1e-5, "x2": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x1 is a small positive float, x2 is negative integer
    input_dict = {"x1": 1e-7, "x2": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x1 is positive infinity, x2 is integer
    input_dict = {"x1": float("inf"), "x2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: x1 is negative infinity, x2 is integer
    input_dict = {"x1": float("-inf"), "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: x1 is NaN, x2 is integer
    input_dict = {"x1": float("nan"), "x2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: x1 is large negative float, x2 is positive integer
    input_dict = {"x1": -9999.99, "x2": 8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.heaviside_7"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_7'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_7'], lib="jax", suffix=7)
