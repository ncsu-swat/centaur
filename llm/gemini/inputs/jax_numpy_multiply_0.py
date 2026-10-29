
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def multiply_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays, identical shape, with negative values
    x = np.array([-2.5, 0.0, 3.1, -4.8], dtype=np.float32)
    y = np.array([1.5, -2.0, 4.0, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D integer arrays
    x = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    y = np.array([[10, 20, 30], [-1, -2, -3]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: Broadcasting 2D (4, 1) and (1, 5)
    x = np.arange(4, dtype=np.float64).reshape(4, 1)
    y = np.arange(5, dtype=np.float64).reshape(1, 5)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar arrays
    x = np.array(7.5, dtype=np.float32)
    y = np.array(-3.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar array and 3D array (broadcasting)
    x = np.array(-2, dtype=np.int64)
    y = np.ones((2, 3, 4), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D float arrays with broadcasting (2, 1, 4) and (1, 3, 4)
    x = np.random.randn(2, 1, 4).astype(np.float32)
    y = np.random.randn(1, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Complex arrays
    x = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64)
    y = np.array([2.0 - 1.0j, 0.5 + 1.5j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: Boolean arrays
    x = np.array([[True, False], [True, True]], dtype=np.bool_)
    y = np.array([[False, True], [True, False]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D array and 1D array broadcasting
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Unsigned integers with 2D shapes
    x = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    y = np.array([[2, 3], [4, 5]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Empty array with matching broadcastable shape
    x = np.empty((0, 5), dtype=np.float32)
    y = np.empty((1, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 12: Float64 array with special values (inf, nan)
    x = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float64)
    y = np.array([0.0, 2.0, -1.0, 5.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.multiply"] = multiply_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.multiply' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.multiply'.")


check_valid('jax.numpy.multiply', generated_inputs['jax.numpy.multiply'], lib="jax", suffix=0)
