
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fill_diagonal_inputs():
    list_of_inputs = []

    # Input 1: Square 2D float32 array
    a = np.zeros((3, 3), dtype=np.float32)
    input_dict = {
        "a": a,
        "val": 1.0,
        "wrap": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Square 2D int32 array, negative float fill value
    a = np.zeros((4, 4), dtype=np.int32)
    input_dict = {
        "a": a,
        "val": -2.5,
        "wrap": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-square 2D float64 array (more columns)
    a = np.zeros((2, 5), dtype=np.float64)
    input_dict = {
        "a": a,
        "val": 0.0,
        "wrap": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Non-square 2D float32 array (more rows)
    a = np.zeros((5, 2), dtype=np.float32)
    input_dict = {
        "a": a,
        "val": 99.9,
        "wrap": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array (all dimensions must be same size)
    a = np.zeros((3, 3, 3), dtype=np.float32)
    input_dict = {
        "a": a,
        "val": 3.14,
        "wrap": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array (all dimensions must be same size)
    a = np.zeros((2, 2, 2, 2), dtype=np.float32)
    input_dict = {
        "a": a,
        "val": -1.0,
        "wrap": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Square 2D int64 array
    a = np.zeros((5, 5), dtype=np.int64)
    input_dict = {
        "a": a,
        "val": 42.0,
        "wrap": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger square 2D float32 array
    a = np.zeros((10, 10), dtype=np.float32)
    input_dict = {
        "a": a,
        "val": -0.001,
        "wrap": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large non-square 2D float64 array
    a = np.zeros((4, 10), dtype=np.float64)
    input_dict = {
        "a": a,
        "val": 123.456,
        "wrap": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array with dimension 4
    a = np.zeros((4, 4, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "val": 0.5,
        "wrap": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fill_diagonal_3"] = fill_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fill_diagonal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fill_diagonal_3'.")


check_valid('jax.numpy.fill_diagonal', generated_inputs['jax.numpy.fill_diagonal_3'], lib="jax", suffix=3)
