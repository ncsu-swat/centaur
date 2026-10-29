
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reduce_precision_inputs():
    list_of_inputs = []

    # Input 1: Float16 equivalent parameters
    input_dict = {
        "operand": 1.0,
        "exponent_bits": 5,
        "mantissa_bits": 10,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Bfloat16 equivalent parameters with negative value
    input_dict = {
        "operand": -2.5,
        "exponent_bits": 8,
        "mantissa_bits": 7,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero operand with small bit widths
    input_dict = {
        "operand": 0.0,
        "exponent_bits": 4,
        "mantissa_bits": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Standard Float32 precision bounds
    input_dict = {
        "operand": 3.1415926535,
        "exponent_bits": 8,
        "mantissa_bits": 23,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Low precision negative fraction
    input_dict = {
        "operand": -0.125,
        "exponent_bits": 2,
        "mantissa_bits": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger positive float with custom bits
    input_dict = {
        "operand": 100.5,
        "exponent_bits": 6,
        "mantissa_bits": 9,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small negative value
    input_dict = {
        "operand": -0.001953125,
        "exponent_bits": 5,
        "mantissa_bits": 5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero mantissa bits (exponent only precision)
    input_dict = {
        "operand": 42.0,
        "exponent_bits": 8,
        "mantissa_bits": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Intermediate precision
    input_dict = {
        "operand": 0.70710678,
        "exponent_bits": 4,
        "mantissa_bits": 7,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger negative float with higher precision mantissa
    input_dict = {
        "operand": -999.0,
        "exponent_bits": 7,
        "mantissa_bits": 15,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Very small positive float
    input_dict = {
        "operand": 1e-4,
        "exponent_bits": 5,
        "mantissa_bits": 8,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Minimal 1-bit exponent and mantissa
    input_dict = {
        "operand": -4.0,
        "exponent_bits": 1,
        "mantissa_bits": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.reduce_precision_2"] = reduce_precision_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_precision_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_precision_2'.")


check_valid('jax.lax.reduce_precision', generated_inputs['jax.lax.reduce_precision_2'], lib="jax", suffix=2)
