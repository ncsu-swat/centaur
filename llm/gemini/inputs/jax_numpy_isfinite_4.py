
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isfinite_inputs():
    list_of_inputs = []

    # Input 1: 1D array of floats with positive, negative, inf, and nan
    x = np.array([1.0, -2.5, np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array of integers
    x = np.array([0, 1, -5, 100, -9999], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of floats with inf
    x = np.array([[1.5, 2.5], [np.inf, 4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array of regular floats
    x = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array of floats with nan and -inf
    x = np.array(
        [[[1.0, 2.0], [np.nan, 4.0]], [[5.0, 6.0], [7.0, -np.inf]]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D boolean array
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array with extreme finite float values
    x = np.array([1e30, -1e30, 0.0, 1e-30, -1e-30], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D array of integers
    x = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Single-element array with finite value
    x = np.array([42.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Single-element array with nan
    x = np.array([np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: 4D array of floats
    x = np.ones((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.isfinite_4"] = isfinite_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isfinite_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isfinite_4'.")


check_valid('jax.numpy.isfinite', generated_inputs['jax.numpy.isfinite_4'], lib="jax", suffix=4)
