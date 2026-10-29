
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def not_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D integer arrays of same shape
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([1, 0, 3, -4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D float arrays of same shape
    x = np.array([[1.0, -2.5], [3.2, 4.0]], dtype=np.float32)
    y = np.array([[1.0, 2.5], [0.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: Broadcasting 2D array and 1D array
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    y = np.array([1, 5, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: Scalar 0-D arrays
    x = np.array(5.0, dtype=np.float64)
    y = np.array(-5.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Boolean arrays
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    y = np.array([[False, False], [True, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D float arrays with broadcasting
    x = np.ones((2, 1, 4), dtype=np.float32)
    y = np.zeros((1, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Complex arrays
    x = np.array([1 + 2j, 3 - 4j, 0 + 0j], dtype=np.complex64)
    y = np.array([1 + 2j, 3 + 4j, 0 + 0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Special values (NaN, Inf, -Inf)
    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float64)
    y = np.array([np.nan, np.inf, np.inf, -0.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Empty arrays
    x = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 4D arrays with matching shapes
    x = np.random.randint(-10, 10, size=(2, 3, 4, 5)).astype(np.int32)
    y = np.random.randint(-10, 10, size=(2, 3, 4, 5)).astype(np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Broadcasting 1D array to 3D array
    x = np.arange(5, dtype=np.int16)
    y = np.zeros((2, 3, 5), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.not_equal_1"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_1'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_1'], lib="jax", suffix=1)
