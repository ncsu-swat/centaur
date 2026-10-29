
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def flatnonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with positive values and zeros
    a = np.array([0, 1, 0, 3, 4, 0, 5], dtype=np.int32)
    input_dict = {
        "a": a,
        "size": 5,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array with negative and positive values, padding needed
    a = np.array([[-1.5, 0.0, 2.3], [0.0, -4.1, 0.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "size": 6,
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int64 array, size smaller than number of nonzeros (truncation)
    a = np.array(
        [[[1, 2], [0, 4]], [[5, 0], [7, 8]]],
        dtype=np.int64,
    )
    input_dict = {
        "a": a,
        "size": 3,
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D boolean array
    a = np.array([True, False, True, True, False], dtype=np.bool_)
    input_dict = {
        "a": a,
        "size": 5,
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of all zeros
    a = np.zeros((3, 3), dtype=np.float32)
    input_dict = {
        "a": a,
        "size": 4,
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array with all non-zero elements
    a = np.ones((2, 4), dtype=np.int32)
    input_dict = {
        "a": a,
        "size": 8,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float64 array with negative numbers and zeros
    a = np.array(
        [
            [[[-1.0, 0.0], [2.0, -3.0]], [[0.0, 0.0], [-5.0, 6.0]]],
            [[[0.0, 7.0], [-8.0, 0.0]], [[9.0, -10.0], [0.0, 11.0]]],
        ],
        dtype=np.float64,
    )
    input_dict = {
        "a": a,
        "size": 12,
        "fill_value": -99.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int16 array with size=0
    a = np.array([-10, 0, 20, -30, 40], dtype=np.int16)
    input_dict = {
        "a": a,
        "size": 0,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 0D scalar array non-zero
    a = np.array(42, dtype=np.int32)
    input_dict = {
        "a": a,
        "size": 2,
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D uint8 array
    a = np.array([[[0, 255], [128, 0]], [[0, 0], [64, 255]]], dtype=np.uint8)
    input_dict = {
        "a": a,
        "size": 6,
        "fill_value": 1000.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D complex64 array
    a = np.array([0 + 0j, 1 - 2j, 0 + 0j, -3 + 4j, 5 + 0j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "size": 4,
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.flatnonzero_3"] = flatnonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flatnonzero_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flatnonzero_3'.")


check_valid('jax.numpy.flatnonzero', generated_inputs['jax.numpy.flatnonzero_3'], lib="jax", suffix=3)
