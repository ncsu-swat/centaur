
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def left_shift_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D arrays, int32
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: 2D arrays, int16
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[2, 3], [4, 5]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 0D arrays (scalar representation as tensor), int64
    x = np.array(5, dtype=np.int64)
    y = np.array(3, dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Broadcasting a scalar-like 1D array to a 2D array, int8
    x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int8)
    y = np.array([1], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Negative values in x, int8, with small shifts
    x = np.array([-1, -2, -3, -4], dtype=np.int8)
    y = np.array([1, 2, 1, 2], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: 1D arrays, int64
    x = np.array([100, 200, 300], dtype=np.int64)
    y = np.array([0, 5, 10], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: 3D arrays, int32
    x = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    y = np.ones((2, 2, 2), dtype=np.int32) * 2
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: Broadcasting compatible shapes (1, 3) and (3, 1), int16
    x = np.array([[1, 2, 3]], dtype=np.int16)
    y = np.array([[1], [2], [3]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: Large integers with int64, shifting by large amounts
    x = np.array([123456789, 987654321], dtype=np.int64)
    y = np.array([10, 20], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: Zero values, int32
    x = np.zeros((4, 4), dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 11: Shift amount as 0, int8
    x = np.array([15, 30, 45], dtype=np.int8)
    y = np.array([0, 0, 0], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.left_shift_1"] = left_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.left_shift_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.left_shift_1'.")


check_valid('jax.numpy.left_shift', generated_inputs['jax.numpy.left_shift_1'], lib="jax", suffix=1)
