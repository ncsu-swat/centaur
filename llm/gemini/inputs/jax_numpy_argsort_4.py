
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def argsort_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, axis=0
    a = np.array([True, False, True, False, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": False,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D boolean array, axis=-1, descending
    a = np.array([False, False, True, True, False, True, False, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": False,
        "descending": True,
        "dtype": np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D boolean array, axis=0
    a = np.array(
        [[True, False, True, False], [False, True, False, True], [True, True, False, False]],
        dtype=bool,
    )
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": True,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D boolean array, axis=1
    a = np.array(
        [[True, True, False], [False, True, False], [False, False, True], [True, False, True]],
        dtype=bool,
    )
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": False,
        "descending": False,
        "dtype": np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D boolean array, axis=-1, all True
    a = np.ones((5, 5), dtype=bool)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": False,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D boolean array, axis=-2, all False
    a = np.zeros((2, 6), dtype=bool)
    input_dict = {
        "a": a,
        "axis": -2,
        "stable": False,
        "descending": True,
        "dtype": np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D boolean array, axis=0
    a = (np.random.randn(2, 3, 4) > 0).astype(bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": False,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D boolean array, axis=1
    a = (np.random.randn(3, 2, 4) > 0).astype(bool)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": False,
        "descending": True,
        "dtype": np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D boolean array, axis=2
    a = (np.random.randn(2, 4, 3) > 0).astype(bool)
    input_dict = {
        "a": a,
        "axis": 2,
        "stable": True,
        "descending": True,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D boolean array, axis=-1
    a = (np.random.randn(4, 2, 3) > 0).astype(bool)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": False,
        "descending": False,
        "dtype": np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D boolean array, axis=3
    a = (np.random.randn(2, 2, 3, 3) > 0).astype(bool)
    input_dict = {
        "a": a,
        "axis": 3,
        "stable": True,
        "descending": False,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 4D boolean array, axis=-3
    a = (np.random.randn(2, 3, 2, 2) > 0).astype(bool)
    input_dict = {
        "a": a,
        "axis": -3,
        "stable": True,
        "descending": True,
        "dtype": np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.argsort_4"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argsort_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argsort_4'.")


check_valid('jax.numpy.argsort', generated_inputs['jax.numpy.argsort_4'], lib="jax", suffix=4)
