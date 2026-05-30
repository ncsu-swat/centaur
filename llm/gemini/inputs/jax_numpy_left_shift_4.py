
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def left_shift_inputs():
    list_of_inputs = []

    # Input 1: Basic scalars (int32)
    x = np.array(5, dtype=np.int32)
    y = np.array(2, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 1D arrays of the same size (int32)
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: Broadcasting 1D array and scalar (int64)
    x = np.array([10, 20, 30], dtype=np.int64)
    y = np.array(3, dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Broadcasting scalar and 1D array (int16)
    x = np.array(7, dtype=np.int16)
    y = np.array([0, 1, 2, 3], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: 2D arrays (int32)
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 3], [4, 5]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Negative values in x (int32)
    x = np.array([-1, -5, -10], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: 8-bit integers (int8)
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([1, 2, 3], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: Unsigned integers (uint32)
    x = np.array([15, 30], dtype=np.uint32)
    y = np.array([2, 4], dtype=np.uint32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: 3D arrays (int32)
    x = np.ones((2, 2, 2), dtype=np.int32)
    y = np.ones((2, 2, 2), dtype=np.int32) * 2
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: Broadcasting 2D and 1D arrays (int32)
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: Shift by 0 (int32)
    x = np.array([12, 34, 56], dtype=np.int32)
    y = np.array([0, 0, 0], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.left_shift_4"] = left_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.left_shift_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.left_shift_4'.")


check_valid('jax.numpy.left_shift', generated_inputs['jax.numpy.left_shift_4'], lib="jax", suffix=4)
