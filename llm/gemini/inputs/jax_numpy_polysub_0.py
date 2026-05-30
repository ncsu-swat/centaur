
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polysub_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays of the same size, integer type
    a1 = np.array([1, 2, 3], dtype=np.int32)
    a2 = np.array([4, 5, 6], dtype=np.int32)
    list_of_inputs.append({"a1": a1, "a2": a2})

    # Input 2: 1D arrays of different sizes, float32
    a1 = np.array([1.5, 2.5], dtype=np.float32)
    a2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"a1": a1, "a2": a2})

    # Input 3: 1D arrays with negative values, float64
    a1 = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    a2 = np.array([1.0, -1.0], dtype=np.float64)
    list_of_inputs.append({"a1": a1, "a2": a2})

    # Input 4: 2D arrays of the same shape, int32
    a1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    a2 = np.array([[5, 6], [7, 8]], dtype=np.int32)
    list_of_inputs.append({"a1": a1, "a2": a2})

    # Input 5: 2D arrays with broadcasting along the first dimension
    a1 = np.array([[2, 3, 1]], dtype=np.int32)
    a2 = np.array([[5, 7, 3], [8, 2, 6]], dtype=np.int32)
    list_of_inputs.append({"a1": a1, "a2": a2})

    # Input 6: 1D arrays, one is very small (size 1)
    a1 = np.array([5, 4, 3, 2, 1], dtype=np.int64)
    a2 = np.array([2], dtype=np.int64)
    list_of_inputs.append({"a1": a1, "a2": a2})

    # Input 7: 3D arrays of the same shape
    a1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    a2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float32)
    list_of_inputs.append({"a1": a1, "a2": a2})

    # Input 8: 2D arrays with broadcasting on the second dimension (last dimension is 1 for a2)
    a1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    a2 = np.array([[1], [2]], dtype=np.float64)
    list_of_inputs.append({"a1": a1, "a2": a2})

    # Input 9: 1D arrays containing zeros, int32
    a1 = np.array([0, 0, 1], dtype=np.int32)
    a2 = np.array([0, 0, 0, 2], dtype=np.int32)
    list_of_inputs.append({"a1": a1, "a2": a2})

    # Input 10: 1D arrays with large values, float32
    a1 = np.array([1e5, 2e5], dtype=np.float32)
    a2 = np.array([5e4], dtype=np.float32)
    list_of_inputs.append({"a1": a1, "a2": a2})

    return list_of_inputs

generated_inputs["jax.numpy.polysub"] = polysub_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polysub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polysub'.")


check_valid('jax.numpy.polysub', generated_inputs['jax.numpy.polysub'], lib="jax", suffix=0)
