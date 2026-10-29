
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def modf_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    input_dict = {
        "x": 4.8,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative float
    input_dict = {
        "x": -3.4,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float representation of an integer
    input_dict = {
        "x": 5.0,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative integer as float
    input_dict = {
        "x": -10.0,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero
    input_dict = {
        "x": 0.0,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative zero
    input_dict = {
        "x": -0.0,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small positive float
    input_dict = {
        "x": 1.25e-5,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large positive float
    input_dict = {
        "x": 123456.789,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative number with magnitude less than 1
    input_dict = {
        "x": -0.875,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mathematical constant approximation
    input_dict = {
        "x": 3.141592653589793,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Positive infinity
    input_dict = {
        "x": float("inf"),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative infinity
    input_dict = {
        "x": float("-inf"),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.modf_2"] = modf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.modf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.modf_2'.")


check_valid('jax.numpy.modf', generated_inputs['jax.numpy.modf_2'], lib="jax", suffix=2)
