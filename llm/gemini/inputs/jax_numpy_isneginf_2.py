
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax._src.numpy.ufuncs as ufuncs

orig_isposneginf = ufuncs._isposneginf
ufuncs._isposneginf = lambda infval, x, out: orig_isposneginf(infval, x, None)

def isneginf_inputs():
    list_of_inputs = []

    # Input 1: Negative infinity
    input_dict = {
        "x": float('-inf'),
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive infinity
    input_dict = {
        "x": float('inf'),
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Positive float
    input_dict = {
        "x": 123.45,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative float
    input_dict = {
        "x": -67.89,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero
    input_dict = {
        "x": 0.0,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative zero
    input_dict = {
        "x": -0.0,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Not a number (NaN)
    input_dict = {
        "x": float('nan'),
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large float
    input_dict = {
        "x": 1e300,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very small negative float
    input_dict = {
        "x": -1e300,
        "out": np.empty((), dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tiny positive float
    input_dict = {
        "x": 1e-300,
        "out": np.empty((), dtype=bool)
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
