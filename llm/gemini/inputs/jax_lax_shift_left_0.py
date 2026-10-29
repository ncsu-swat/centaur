
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def shift_left_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays of same shape
    x = np.array([1, 2, 4, 8, 16], dtype=np.int32)
    y = np.array([1, 2, 1, 0, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int32 arrays with broadcasting
    x = np.array([[1], [3], [5]], dtype=np.int32)
    y = np.array([[0, 1, 2, 3]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D int32 arrays of same shape
    x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    y = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 1D int32 arrays with negative values in x
    x = np.array([-1, -4, 5, -8], dtype=np.int32)
    y = np.array([1, 2, 1, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D int32 arrays with broadcasting
    x = np.ones((2, 1, 4), dtype=np.int32)
    y = np.zeros((2, 3, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 0D scalar int64 arrays
    x = np.array(7, dtype=np.int64)
    y = np.array(3, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D int32 x and 0D scalar int32 y
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array(2, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D int64 arrays
    x = np.array([1, 3, 7, 15, 31, 63], dtype=np.int64)
    y = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D int32 arrays with broadcasting
    x = np.arange(6, dtype=np.int32).reshape(1, 2, 1, 3)
    y = np.ones((2, 2, 4, 3), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 1D int64 arrays with higher shifts
    x = np.array([100, 200, 300, 400], dtype=np.int64)
    y = np.array([0, 4, 8, 12], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 2D int64 arrays with row broadcasting
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    y = np.array([[1, 2, 3, 4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.shift_left"] = shift_left_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.shift_left' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.shift_left'.")


check_valid('jax.lax.shift_left', generated_inputs['jax.lax.shift_left'], lib="jax", suffix=0)
