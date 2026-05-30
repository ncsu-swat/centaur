
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_isposinf_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    input_dict = {
        "x": 5.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive infinity
    input_dict = {
        "x": float('inf'),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative infinity
    input_dict = {
        "x": float('-inf'),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NaN
    input_dict = {
        "x": float('nan'),
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative float
    input_dict = {
        "x": -10.5,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero
    input_dict = {
        "x": 0.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very large float
    input_dict = {
        "x": 1.7976931348623157e+308,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small float
    input_dict = {
        "x": 2.2250738585072014e-308,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative zero
    input_dict = {
        "x": -0.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Arbitrary positive float
    input_dict = {
        "x": 12345.6789,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isposinf_2"] = jax_numpy_isposinf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isposinf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isposinf_2'.")


check_valid('jax.numpy.isposinf', generated_inputs['jax.numpy.isposinf_2'], lib="jax", suffix=2)
