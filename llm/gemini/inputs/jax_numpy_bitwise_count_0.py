
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_count_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive int32
    x = np.array([64, 32, 31, 20], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with negative numbers (int32)
    x = np.array([-16, -7, 7, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of int64
    x = np.array([[2, -7], [-9, 7]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar array (int32)
    x = np.array(-42, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D array of powers of 2 (int64)
    x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D array of int32
    x = np.array(
        [[[1, -2], [3, -4]], [[5, -6], [7, -8]], [[9, -10], [11, -12]]],
        dtype=np.int32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array of alternating bit patterns (int32)
    x = np.array([1431655765, 0, -1, 1073741823], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D array of int32
    x = np.ones((2, 2, 3, 2), dtype=np.int32) * 255
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D array of int64 with large values
    x = np.array([2**30, 2**40, -(2**30), -(2**40)], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D array of zeros and ones (int32)
    x = np.zeros((3, 3), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 2D array reshaped from range (int64)
    x = np.arange(-6, 6, dtype=np.int64).reshape(3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_count"] = bitwise_count_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_count' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_count'.")


check_valid('jax.numpy.bitwise_count', generated_inputs['jax.numpy.bitwise_count'], lib="jax", suffix=0)
