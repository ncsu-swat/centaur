
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sub_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays of int32 (same shape)
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D arrays of int32 (same shape)
    x = np.array([[5, 10], [15, 20]], dtype=np.int32)
    y = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 1D arrays of int64 with negative values (same shape)
    x = np.array([-10, 0, 10], dtype=np.int64)
    y = np.array([5, -5, 5], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Scalar arrays (0D) of int32
    x = np.array(42, dtype=np.int32)
    y = np.array(12, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Broadcasting 2D arrays with the same number of dimensions
    x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64) # Shape (2, 3)
    y = np.array([[1, 2, 3]], dtype=np.int64) # Shape (1, 3)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Broadcasting 2D array and a scalar (0D array)
    x = np.array([[10, 20], [30, 40]], dtype=np.int32) # Shape (2, 2)
    y = np.array(5, dtype=np.int32) # Shape ()
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: 3D arrays of int16 (same shape)
    x = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    y = np.ones((2, 3, 4), dtype=np.int16) * 5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: 1D arrays of int8 (same shape)
    x = np.array([100, -50, 120], dtype=np.int8)
    y = np.array([20, 30, -10], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: 4D arrays of int32 with random values (same shape)
    x = np.random.randint(-50, 50, size=(2, 2, 2, 2), dtype=np.int32)
    y = np.random.randint(-50, 50, size=(2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: Larger 2D arrays of int64 (same shape)
    x = np.random.randint(-1000, 1000, size=(10, 10), dtype=np.int64)
    y = np.random.randint(-1000, 1000, size=(10, 10), dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.lax.sub_2"] = sub_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sub_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sub_2'.")


check_valid('jax.lax.sub', generated_inputs['jax.lax.sub_2'], lib="jax", suffix=2)
