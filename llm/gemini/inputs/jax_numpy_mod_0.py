
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def mod_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays
    x1 = np.array([10, 20, 30, 40], dtype=np.int32)
    x2 = np.array([3, 7, 9, 11], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 2D int32 arrays
    x1 = np.array([[10, 15], [20, 25]], dtype=np.int32)
    x2 = np.array([[3, 4], [5, 6]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: 1D int32 arrays with negative values in x1
    x1 = np.array([-10, -20, 30, -40], dtype=np.int32)
    x2 = np.array([3, 7, 9, 11], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: 1D int32 arrays with negative values in x2
    x1 = np.array([10, 20, 30, 40], dtype=np.int32)
    x2 = np.array([-3, -7, -9, -11], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: Float32 arrays
    x1 = np.array([5.5, 10.25, 15.75], dtype=np.float32)
    x2 = np.array([2.0, 3.5, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: Float32 arrays with negative values
    x1 = np.array([-5.5, 10.25, -15.75], dtype=np.float32)
    x2 = np.array([2.0, -3.5, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: Broadcasting 2D array and 1D array
    x1 = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    x2 = np.array([3, 4, 5], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: Broadcasting 2D array and a 1D scalar-like array
    x1 = np.array([[10, 20], [30, 40]], dtype=np.int32)
    x2 = np.array([3], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Int64 3D arrays
    x1 = np.arange(1, 9, dtype=np.int64).reshape((2, 2, 2))
    x2 = np.array([[[3, 3], [3, 3]], [[3, 3], [3, 3]]], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: Float64 arrays with large values
    x1 = np.array([1e10, 2.5e10], dtype=np.float64)
    x2 = np.array([3.0, 7.0], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 11: Scalar arrays (0-D)
    x1 = np.array(15, dtype=np.int32)
    x2 = np.array(4, dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.mod"] = mod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.mod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.mod'.")


check_valid('jax.numpy.mod', generated_inputs['jax.numpy.mod'], lib="jax", suffix=0)
