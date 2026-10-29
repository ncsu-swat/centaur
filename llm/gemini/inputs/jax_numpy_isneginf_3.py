
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isneginf_inputs():
    list_of_inputs = []

    # Input 1: zero
    input_dict = {
        "x": 0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: small negative integer
    input_dict = {
        "x": -1,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: small positive integer
    input_dict = {
        "x": 1,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive integer
    input_dict = {
        "x": 42,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative integer
    input_dict = {
        "x": -100,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: single digit positive integer
    input_dict = {
        "x": 7,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large negative integer
    input_dict = {
        "x": -999999,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: int32 max
    input_dict = {
        "x": 2147483647,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int32 min
    input_dict = {
        "x": -2147483648,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large positive integer
    input_dict = {
        "x": 1000000,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: moderate negative integer
    input_dict = {
        "x": -42,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isneginf_3"] = isneginf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isneginf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isneginf_3'.")


check_valid('jax.numpy.isneginf', generated_inputs['jax.numpy.isneginf_3'], lib="jax", suffix=3)
