
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: 1D integer arrays of same shape
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 6, 7, 8], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D integer arrays of same shape
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[5, 6], [7, 8]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: Boolean arrays
    x = np.array([True, False, True, False], dtype=bool)
    y = np.array([False, True, True, False], dtype=bool)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: Broadcasting with 1D and 2D arrays
    x = np.array([[1], [2], [3]], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: Negative integer values
    x = np.array([-1, -2, -3], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: Unsigned integer types (uint8)
    x = np.array([255, 0, 127], dtype=np.uint8)
    y = np.array([0, 255, 128], dtype=np.uint8)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 0D array (scalar) and 1D array
    x = np.array(5, dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: High-dimensional arrays (3D)
    x = np.ones((2, 3, 4), dtype=np.int32) * 3
    y = np.ones((2, 3, 4), dtype=np.int32) * 5
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Int16 dtype arrays
    x = np.array([100, 200, 300], dtype=np.int16)
    y = np.array([400, 500, 600], dtype=np.int16)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: Boolean broadcasting
    x = np.array([[True], [False]], dtype=bool)
    y = np.array([False, True], dtype=bool)
    list_of_inputs.append({"x": x, "y": y})

    # Input 11: Larger array size with random integers
    x = np.random.randint(0, 100, size=(10, 10), dtype=np.int32)
    y = np.random.randint(0, 100, size=(10, 10), dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_or_1"] = bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_or_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_or_1'.")


check_valid('jax.numpy.bitwise_or', generated_inputs['jax.numpy.bitwise_or_1'], lib="jax", suffix=1)
