
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def le_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, comparing with 0
    x = np.array([-5, -2, 0, 1, 3, 7], dtype=np.int32)
    y = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int32 array (3, 4), comparing with positive integer
    x = np.array(
        [[1, 5, 3, 8], [9, 2, 4, 6], [0, 7, 10, -1]], dtype=np.int32
    )
    y = 5
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 3D int32 array (2, 2, 3), comparing with negative integer
    x = np.array(
        [
            [[-15, -10, -5], [-2, 0, 4]],
            [[8, -12, -20], [-1, 3, 9]],
        ],
        dtype=np.int32,
    )
    y = -10
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 1D int32 array, comparing with 1
    x = np.array([-128, -50, 0, 1, 2, 127], dtype=np.int32)
    y = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar array, comparing with exact equal integer
    x = np.array(4, dtype=np.int32)
    y = 4
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D int32 array with shape (2, 1, 3, 2), comparing with 100
    x = np.arange(12, dtype=np.int32).reshape(2, 1, 3, 2) * 10
    y = 100
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D int32 array (4, 4), comparing with -1
    x = np.array(
        [
            [-5, -4, -3, -2],
            [-1, 0, 1, 2],
            [3, 4, 5, 6],
            [-10, -20, 30, 40],
        ],
        dtype=np.int32,
    )
    y = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D int32 array with large integers, comparing with large integer
    x = np.array(
        [1000000, 5000000, 10000000, -1000000, 0],
        dtype=np.int32,
    )
    y = 5000000
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D int32 array with shape (1, 6), comparing with 3
    x = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
    y = 3
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Empty 1D int32 array
    x = np.array([], dtype=np.int32)
    y = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D int32 array (3, 1, 2), comparing with -50
    x = np.array(
        [[[-60, -50]], [[-40, -30]], [[0, 50]]],
        dtype=np.int32,
    )
    y = -50
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 12: 5D int32 array (1, 2, 1, 2, 1), comparing with 2
    x = np.array(
        [[[[[1]], [[2]]]], [[[[3]], [[4]]]]],
        dtype=np.int32,
    )
    y = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.le_6"] = le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_6'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_6'], lib="jax", suffix=6)
