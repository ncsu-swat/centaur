
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def maximum_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float arrays with positive/negative values
    x = np.array([1.5, -2.0, 3.5, -4.0], dtype=np.float32)
    y = np.array([-1.0, 2.0, 3.0, -5.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D Integer arrays
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[4, 3], [2, 1]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: Broadcasting 2D and 1D arrays
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: Broadcasting with singular dimensions
    x = np.array([[1], [2], [3]], dtype=np.float32)
    y = np.array([[4, 5, 6]], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: Float64 arrays containing NaN values
    x = np.array([np.nan, 2.0, 3.0], dtype=np.float64)
    y = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 3D random arrays of float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 4D arrays with broadcasting compatibility
    x = np.random.randn(1, 5, 1, 5).astype(np.float32)
    y = np.random.randn(2, 1, 2, 1).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: Boolean arrays
    x = np.array([True, False, True], dtype=bool)
    y = np.array([False, False, True], dtype=bool)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: 0-dimensional arrays (scalars as tensors)
    x = np.array(5.0, dtype=np.float32)
    y = np.array(10.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: Int64 arrays
    x = np.array([-10, 20, -30], dtype=np.int64)
    y = np.array([0, 0, 0], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.maximum"] = maximum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.maximum'.")


check_valid('jax.numpy.maximum', generated_inputs['jax.numpy.maximum'], lib="jax", suffix=0)
