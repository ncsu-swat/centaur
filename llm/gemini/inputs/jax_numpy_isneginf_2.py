
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isneginf_inputs():
    list_of_inputs = []

    # Input 1: Negative infinity
    input_dict = {
        "x": float("-inf"),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive infinity
    input_dict = {
        "x": float("inf"),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NaN
    input_dict = {
        "x": float("nan"),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero
    input_dict = {
        "x": 0.0,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative zero
    input_dict = {
        "x": -0.0,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Positive finite float
    input_dict = {
        "x": 1.0,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative finite float
    input_dict = {
        "x": -1.0,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Arbitrary decimal float
    input_dict = {
        "x": 3.141592653589793,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large negative float
    input_dict = {
        "x": -1e300,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Small positive float (near zero)
    input_dict = {
        "x": 1e-300,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Minimum finite float64
    input_dict = {
        "x": float(np.finfo(np.float64).min),
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative non-integer float
    input_dict = {
        "x": -42.75,
        "out": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isneginf_2"] = isneginf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isneginf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isneginf_2'.")


check_valid('jax.numpy.isneginf', generated_inputs['jax.numpy.isneginf_2'], lib="jax", suffix=2)
