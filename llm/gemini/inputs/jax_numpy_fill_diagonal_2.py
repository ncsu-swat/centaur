
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fill_diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D square int32 array
    input_dict = {
        "a": np.zeros((3, 3), dtype=np.int32),
        "val": 1,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D square float32 array with negative val
    input_dict = {
        "a": np.ones((5, 5), dtype=np.float32),
        "val": -5,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D rectangular (more columns than rows) int64 array
    input_dict = {
        "a": np.zeros((3, 6), dtype=np.int64),
        "val": 42,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D rectangular (more rows than columns) float64 array
    input_dict = {
        "a": np.zeros((6, 3), dtype=np.float64),
        "val": 0,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array with all equal dimensions
    input_dict = {
        "a": np.zeros((2, 2, 2), dtype=np.int32),
        "val": 7,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array with negative val
    input_dict = {
        "a": np.ones((3, 3, 3), dtype=np.float32),
        "val": -1,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D equal dimension int16 array
    input_dict = {
        "a": np.zeros((2, 2, 2, 2), dtype=np.int16),
        "val": 100,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Minimal 1x1 2D uint8 array
    input_dict = {
        "a": np.zeros((1, 1), dtype=np.uint8),
        "val": 255,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array with large negative integer
    input_dict = {
        "a": np.random.randn(4, 7).astype(np.float32),
        "val": -1000,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex64 array
    input_dict = {
        "a": np.zeros((4, 4), dtype=np.complex64),
        "val": 3,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D int8 array
    input_dict = {
        "a": np.zeros((5, 2), dtype=np.int8),
        "val": -12,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fill_diagonal_2"] = fill_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fill_diagonal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fill_diagonal_2'.")


check_valid('jax.numpy.fill_diagonal', generated_inputs['jax.numpy.fill_diagonal_2'], lib="jax", suffix=2)
