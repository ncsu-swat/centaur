
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logical_or_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean arrays of same shape
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D boolean arrays with broadcasting (3, 1) and (1, 4)
    x = np.array([[True], [False], [True]], dtype=np.bool_)
    y = np.array([[False, True, False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 1D integer arrays containing negative, zero, and positive values
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    y = np.array([0, 1, 0, -1, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D float32 arrays including zeros and non-zeros
    x = np.array([[0.0, 1.5], [-0.5, 0.0]], dtype=np.float32)
    y = np.array([[0.0, 0.0], [2.3, -4.1]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar arrays
    x = np.array(0, dtype=np.int64)
    y = np.array(1, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D boolean arrays of shape (2, 3, 4)
    x = np.random.choice([True, False], size=(2, 3, 4)).astype(np.bool_)
    y = np.random.choice([True, False], size=(2, 3, 4)).astype(np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Broadcasting 2D and 1D int64 arrays
    x = np.array([[0, 2, 0], [1, 0, -3]], dtype=np.int64)
    y = np.array([0, 1, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Float64 arrays with special values like inf and nan
    x = np.array([np.nan, 0.0, np.inf, -np.inf], dtype=np.float64)
    y = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Complex numbers
    x = np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 2.0j], dtype=np.complex64)
    y = np.array([0.0 + 0.0j, 0.0 + 0.0j, 1.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 4D array and broadcastable 1D array
    x = np.zeros((2, 1, 3, 4), dtype=np.uint8)
    y = np.ones((4,), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Empty arrays
    x = np.array([], dtype=np.bool_)
    y = np.array([], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.logical_or"] = logical_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logical_or' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logical_or'.")


check_valid('jax.numpy.logical_or', generated_inputs['jax.numpy.logical_or'], lib="jax", suffix=0)
