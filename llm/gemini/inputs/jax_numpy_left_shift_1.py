
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def left_shift_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays with same shape
    x = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    y = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D array and 0D scalar array broadcasting
    x = np.array([10, 20, 30, 40], dtype=np.int64)
    y = np.array(2, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D int32 arrays with negative values in x
    x = np.array([[-1, 2], [-5, 8]], dtype=np.int32)
    y = np.array([[1, 2], [3, 0]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D broadcasting (row vector shifted by column vector)
    x = np.array([[1, 2, 3, 4]], dtype=np.int32)
    y = np.array([[0], [1], [2]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 1D int64 array with zero shift
    x = np.array([1, 3, 7, 15], dtype=np.int64)
    y = np.array([0, 0, 0, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D int32 arrays
    x = np.arange(8, dtype=np.int32).reshape((2, 2, 2))
    y = np.ones((2, 2, 2), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 0D scalars
    x = np.array(16, dtype=np.int32)
    y = np.array(3, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D int32 arrays with larger values
    x = np.array([1000, 20000, 300000], dtype=np.int32)
    y = np.array([2, 4, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D int64 arrays with negative values in x
    x = np.array([[-8, -4], [0, 4]], dtype=np.int64)
    y = np.array([[1, 2], [0, 1]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 4D int32 arrays
    x = np.arange(24, dtype=np.int32).reshape((2, 3, 1, 4))
    y = np.ones((1, 3, 2, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

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
