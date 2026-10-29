
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, size equals number of non-zero elements
    a = np.array([0, 1, 0, 2, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "size": 3,
        "fill_value": (-1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array, size larger than non-zero elements (padding)
    a = np.array([0.0, -1.5, 0.0, 3.2, 0.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "size": 6,
        "fill_value": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int64 array, size smaller than non-zero elements (truncation)
    a = np.array([[1, 0, 2], [0, -3, 4]], dtype=np.int64)
    input_dict = {
        "a": a,
        "size": 2,
        "fill_value": (-1, -1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, padded with custom fill values
    a = np.array([[0.0, 0.0], [5.5, 0.0], [0.0, -2.1]], dtype=np.float64)
    input_dict = {
        "a": a,
        "size": 5,
        "fill_value": (99, 99)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D boolean array
    a = np.array([[True, False, True], [False, False, True]], dtype=np.bool_)
    input_dict = {
        "a": a,
        "size": 4,
        "fill_value": (0, 0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array
    a = np.zeros((2, 3, 4), dtype=np.float32)
    a[0, 1, 2] = 1.0
    a[1, 2, 3] = -4.5
    input_dict = {
        "a": a,
        "size": 4,
        "fill_value": (-1, -1, -1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int32 array, size 0 (empty result)
    a = np.ones((2, 2, 2), dtype=np.int32)
    input_dict = {
        "a": a,
        "size": 0,
        "fill_value": (0, 0, 0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D uint8 array
    a = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    a[0, 0, 1, 1] = 255
    a[1, 1, 0, 0] = 128
    input_dict = {
        "a": a,
        "size": 3,
        "fill_value": (255, 255, 255, 255)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D complex64 array with non-zero elements
    a = np.array([0.0 + 0.0j, 1.0 + 2.0j, 0.0 + 0.0j, -3.0 + 0.0j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "size": 5,
        "fill_value": (-1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int16 array with all elements non-zero
    a = np.array([[-10, 20], [30, -40]], dtype=np.int16)
    input_dict = {
        "a": a,
        "size": 4,
        "fill_value": (100, 100)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D int8 array with negative fill values
    a = np.array([[[0, -1], [2, 0]], [[0, 0], [5, -6]]], dtype=np.int8)
    input_dict = {
        "a": a,
        "size": 6,
        "fill_value": (-1, 0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nonzero_2"] = nonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nonzero_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nonzero_2'.")


check_valid('jax.numpy.nonzero', generated_inputs['jax.numpy.nonzero_2'], lib="jax", suffix=2)
