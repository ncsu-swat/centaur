
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def shift_left_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int32 arrays of same shape
    x = np.array([1, 2, 4, 8], dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D int64 arrays of same shape
    x = np.array([[10, 20], [30, 40]], dtype=np.int64)
    y = np.array([[1, 2], [3, 4]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: Another 1D int32 array of same shape
    x = np.array([5, 10, 15], dtype=np.int32)
    y = np.array([2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: 2D broadcasting with SAME number of dimensions (2D and 2D)
    x = np.array([[2, 4], [6, 8]], dtype=np.int32)
    y = np.array([[1, 2]], dtype=np.int32)  # shape (1, 2)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Negative integer values with 2D matching shape
    x = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    y = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: 2D array and 0-D scalar array (y is scalar)
    x = np.array([[5, 6], [7, 8]], dtype=np.int32)
    y = np.array(2, dtype=np.int32)  # shape ()
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: 3D int32 arrays of same shape
    x = np.ones((2, 2, 2), dtype=np.int32)
    y = np.zeros((2, 2, 2), dtype=np.int32) + 3
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: Broadcasting with shape (1, 5) and (5, 1) in int32
    x = np.arange(5, dtype=np.int32).reshape(1, 5)
    y = np.arange(5, dtype=np.int32).reshape(5, 1)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: 1D int64 arrays of same shape
    x = np.array([128, 64, 32], dtype=np.int64)
    y = np.array([1, 2, 3], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: Larger shifts on 1D int32 arrays of same shape
    x = np.array([1, 1], dtype=np.int32)
    y = np.array([30, 31], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: 3D array broadcasting with 3D array (shape 3,3,3 and 1,1,3)
    x = np.random.randint(0, 100, size=(3, 3, 3), dtype=np.int32)
    y = np.array([[[1, 2, 3]]], dtype=np.int32)  # shape (1, 1, 3)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 12: 0-D scalar array and 1D array (x is scalar)
    x = np.array(3, dtype=np.int32)  # shape ()
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

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
