
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def right_shift_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D arrays, int32
    x1 = np.array([2, 4, 8, 16, 32], dtype=np.int32)
    x2 = np.array([1, 1, 2, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 2D arrays, int32
    x1 = np.array([[10, 20], [30, 40]], dtype=np.int32)
    x2 = np.array([[1, 2], [1, 2]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: Broadcasting 1D array with a scalar-like 0D array, int32
    x1 = np.array([128, 64, 32], dtype=np.int32)
    x2 = np.array(2, dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: Broadcasting 3D array and 1D array, int32
    x1 = np.ones((2, 3, 4), dtype=np.int32) * 120
    x2 = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: Large int64 elements with shift values, int64
    x1 = np.array([2**40, 2**45, 2**50], dtype=np.int64)
    x2 = np.array([10, 20, 30], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: 0D arrays (scalars), int32
    x1 = np.array(255, dtype=np.int32)
    x2 = np.array(4, dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: Higher-dimensional 4D arrays, int32
    x1 = np.random.randint(0, 1000, size=(2, 2, 2, 2), dtype=np.int32)
    x2 = np.random.randint(0, 5, size=(2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: Mixed dimensions row-wise broadcasting, int32
    x1 = np.array([[1024, 2048, 4096]], dtype=np.int32)
    x2 = np.array([[2], [4]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Zero shifts (should keep original values), int32
    x1 = np.array([5, 10, 15], dtype=np.int32)
    x2 = np.array([0, 0, 0], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: Boundary values - shifting to 0, int32
    x1 = np.array([1, 3, 7], dtype=np.int32)
    x2 = np.array([4, 4, 4], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 11: Empty arrays, int32
    x1 = np.array([], dtype=np.int32).reshape(0, 5)
    x2 = np.array([], dtype=np.int32).reshape(0, 5)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.right_shift_1"] = right_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.right_shift_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.right_shift_1'.")


check_valid('jax.numpy.right_shift', generated_inputs['jax.numpy.right_shift_1'], lib="jax", suffix=1)
