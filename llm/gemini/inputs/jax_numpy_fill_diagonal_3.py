
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fill_diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D square float32 matrix, positive float val
    input_dict = {
        "a": np.zeros((4, 4), dtype=np.float32),
        "val": 1.5,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D square float64 matrix, negative float val
    input_dict = {
        "a": np.ones((5, 5), dtype=np.float64),
        "val": -3.14159,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D rectangular matrix with more columns than rows
    input_dict = {
        "a": np.zeros((3, 7), dtype=np.float32),
        "val": 2.0,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D rectangular matrix with more rows than columns
    input_dict = {
        "a": np.zeros((6, 2), dtype=np.float32),
        "val": 0.0,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D cube array (all dims must be equal for ndim >= 3)
    input_dict = {
        "a": np.zeros((3, 3, 3), dtype=np.float32),
        "val": 7.5,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D hypercube array
    input_dict = {
        "a": np.zeros((2, 2, 2, 2), dtype=np.float64),
        "val": -0.5,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int32 array filled with float
    input_dict = {
        "a": np.arange(9, dtype=np.int32).reshape((3, 3)),
        "val": 42.0,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int64 rectangular array
    input_dict = {
        "a": np.ones((4, 5), dtype=np.int64),
        "val": -10.0,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D boolean array
    input_dict = {
        "a": np.zeros((3, 3), dtype=bool),
        "val": 1.0,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex64 array
    input_dict = {
        "a": np.zeros((4, 4), dtype=np.complex64),
        "val": 5.0,
        "wrap": False,
        "inplace": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D cube with larger size
    input_dict = {
        "a": np.random.randn(4, 4, 4).astype(np.float32),
        "val": -99.9,
        "wrap": False,
        "inplace": False,
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
