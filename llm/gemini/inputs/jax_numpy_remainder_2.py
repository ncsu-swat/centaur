
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def remainder_inputs():
    list_of_inputs = []

    # Input 1: Basic scalar integers
    x1 = np.int32(10)
    x2 = np.int32(3)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative dividend scalar
    x1 = np.int64(-10)
    x2 = np.int64(3)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative divisor scalar
    x1 = np.int32(10)
    x2 = np.int32(-3)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D arrays of integers
    x1 = np.array([5, -3, 9, 12], dtype=np.int32)
    x2 = np.array([2, 4, -5, 7], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays of integers
    x1 = np.array([[10, 20], [-30, 40]], dtype=np.int64)
    x2 = np.array([[3, 7], [8, -9]], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting 1D array with scalar
    x1 = np.array([10, 15, 22, -35], dtype=np.int32)
    x2 = np.int32(6)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting 2D array with 1D array
    x1 = np.array([[5, 12, 18], [24, 31, 45]], dtype=np.int32)
    x2 = np.array([3, 5, 7], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D arrays with different integer dtypes (int16 and int8)
    x1 = np.array([100, 150, 200], dtype=np.int16)
    x2 = np.array([13, 17, 19], dtype=np.int8)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D arrays
    x1 = np.ones((2, 2, 2), dtype=np.int32) * 17
    x2 = np.ones((2, 2, 2), dtype=np.int32) * 5
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger integer values
    x1 = np.array([1000000000, -2000000000], dtype=np.int64)
    x2 = np.array([3000000, 7000000], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Broadcasting column and row vectors
    x1 = np.array([[10], [20], [30]], dtype=np.int32)
    x2 = np.array([3, 4, 7], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.remainder_2"] = remainder_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.remainder_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.remainder_2'.")


check_valid('jax.numpy.remainder', generated_inputs['jax.numpy.remainder_2'], lib="jax", suffix=2)
