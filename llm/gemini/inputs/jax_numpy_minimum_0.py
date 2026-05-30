
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def minimum_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays of same shape, float32
    x = np.array([1.5, -2.0, 3.7, 0.0], dtype=np.float32)
    y = np.array([-1.0, 2.5, 3.7, -0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 1D arrays of same shape, int32
    x = np.array([10, -5, 20, 0], dtype=np.int32)
    y = np.array([5, -10, 20, 3], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 2D arrays, float64
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[0.0, 3.0], [2.0, 5.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: Broadcasting, 2D and 1D arrays
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([2, 2, 2], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: Broadcasting, 3D and 1D arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(4).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: Arrays with NaNs, float32
    x = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    y = np.array([2.0, 2.0, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 0D arrays (scalars as arrays)
    x = np.array(5.0, dtype=np.float32)
    y = np.array(3.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: High-dimensional arrays (4D)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    y = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Boolean arrays (logical AND behavior)
    x = np.array([True, False, True, False], dtype=bool)
    y = np.array([True, True, False, False], dtype=bool)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: Broadcasting with shape (1, 5) and (5, 1)
    x = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    y = np.array([[5], [4], [3], [2], [1]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.minimum"] = minimum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.minimum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.minimum'.")


check_valid('jax.numpy.minimum', generated_inputs['jax.numpy.minimum'], lib="jax", suffix=0)
