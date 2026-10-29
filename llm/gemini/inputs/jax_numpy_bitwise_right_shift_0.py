
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_right_shift_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays with same shape
    x1 = np.array([16, 32, 64, 128], dtype=np.int32)
    x2 = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D int64 arrays with negative values in x1
    x1 = np.array([[-16, -32], [64, 128]], dtype=np.int64)
    x2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Scalar (0-D array) inputs
    x1 = np.array(1024, dtype=np.int32)
    x2 = np.array(5, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting 1D to 2D with int32
    x1 = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    x2 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: Broadcasting scalar (0-D) shift to 3D int32 array
    x1 = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    x2 = np.array(2, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: int64 arrays with zeros
    x1 = np.array([0, 255, 1024, 65536], dtype=np.int64)
    x2 = np.array([0, 0, 1, 8], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D int32 arrays
    x1 = np.ones((2, 2, 2, 2), dtype=np.int32) * 16
    x2 = np.ones((2, 2, 2, 2), dtype=np.int32) * 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: int32 arrays with negative numbers and zero shift
    x1 = np.array([-128, -64, -1, 0, 63, 127], dtype=np.int32)
    x2 = np.array([1, 2, 0, 0, 3, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Broadcasting column vector and row vector
    x1 = np.array([[8], [16], [32]], dtype=np.int32)
    x2 = np.array([[1, 2, 3]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: int64 large values and shift amounts
    x1 = np.array([2**40, 2**35, 2**30], dtype=np.int64)
    x2 = np.array([10, 20, 30], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 3D int64 arrays with broadcasting across dimensions
    x1 = np.arange(16, dtype=np.int64).reshape((4, 1, 4))
    x2 = np.array([[[1], [2], [3]]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_right_shift"] = bitwise_right_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_right_shift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_right_shift'.")


check_valid('jax.numpy.bitwise_right_shift', generated_inputs['jax.numpy.bitwise_right_shift'], lib="jax", suffix=0)
