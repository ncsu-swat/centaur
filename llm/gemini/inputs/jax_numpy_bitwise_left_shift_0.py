
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_left_shift_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays
    x = np.array([1, 2, 4, 8], dtype=np.int32)
    y = np.array([1, 2, 1, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int64 arrays with negative values
    x = np.array([[-1, 2], [-5, 10]], dtype=np.int64)
    y = np.array([[2, 1], [0, 3]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalar arrays (int32)
    x = np.array(7, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: Broadcasting 2D and 1D (int32)
    x = np.array([[1], [3], [5]], dtype=np.int32)
    y = np.array([0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 1D int64 arrays
    x = np.array([3, 7, 15, 31], dtype=np.int64)
    y = np.array([1, 2, 3, 4], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D int32 arrays
    x = np.ones((2, 3, 4), dtype=np.int32)
    y = np.full((2, 3, 4), 2, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 1D int32 arrays with negative numbers
    x = np.array([-16, -8, 0, 8, 16], dtype=np.int32)
    y = np.array([1, 2, 0, 1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 2D int64 arrays
    x = np.array([[100, 200], [300, 400]], dtype=np.int64)
    y = np.array([[1, 2], [3, 4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 0D scalar arrays (int64)
    x = np.array(-5, dtype=np.int64)
    y = np.array(2, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 4D int32 arrays with broadcasting
    x = np.arange(12, dtype=np.int32).reshape(1, 3, 2, 2)
    y = np.array([[[[1]]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_left_shift"] = bitwise_left_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_left_shift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_left_shift'.")


check_valid('jax.numpy.bitwise_left_shift', generated_inputs['jax.numpy.bitwise_left_shift'], lib="jax", suffix=0)
