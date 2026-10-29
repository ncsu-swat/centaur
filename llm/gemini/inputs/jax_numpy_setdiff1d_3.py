
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def setdiff1d_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D integer arrays
    input_dict = {
        "ar1": np.array([1, 2, 3, 4], dtype=np.int32),
        "ar2": np.array([3, 4, 5, 6], dtype=np.int32),
        "assume_unique": False,
        "size": 4,
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Arrays with duplicates, assume_unique=False
    input_dict = {
        "ar1": np.array([1, 2, 2, 3, 3, 4], dtype=np.int64),
        "ar2": np.array([2, 3, 3], dtype=np.int64),
        "assume_unique": False,
        "size": 3,
        "fill_value": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Unique elements with assume_unique=True
    input_dict = {
        "ar1": np.array([10, 20, 30, 40], dtype=np.int32),
        "ar2": np.array([20, 40], dtype=np.int32),
        "assume_unique": True,
        "size": 2,
        "fill_value": 999,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative integers
    input_dict = {
        "ar1": np.array([-5, -3, -1, 0, 2], dtype=np.int32),
        "ar2": np.array([-3, 2, 4], dtype=np.int32),
        "assume_unique": True,
        "size": 5,
        "fill_value": -99,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Truncated output (size smaller than diff)
    input_dict = {
        "ar1": np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
        "ar2": np.array([10, 20], dtype=np.int32),
        "assume_unique": True,
        "size": 2,
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Disjoint arrays
    input_dict = {
        "ar1": np.array([100, 200, 300], dtype=np.int64),
        "ar2": np.array([400, 500, 600], dtype=np.int64),
        "assume_unique": False,
        "size": 6,
        "fill_value": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty ar1
    input_dict = {
        "ar1": np.array([], dtype=np.int32),
        "ar2": np.array([1, 2, 3], dtype=np.int32),
        "assume_unique": True,
        "size": 2,
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty ar2
    input_dict = {
        "ar1": np.array([5, 6, 7], dtype=np.int32),
        "ar2": np.array([], dtype=np.int32),
        "assume_unique": False,
        "size": 5,
        "fill_value": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multi-dimensional arrays (flattened by setops)
    input_dict = {
        "ar1": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "ar2": np.array([[2, 4], [6, 8]], dtype=np.int32),
        "assume_unique": False,
        "size": 4,
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complete overlap (diff is empty)
    input_dict = {
        "ar1": np.array([1, 2, 3], dtype=np.int32),
        "ar2": np.array([1, 2, 3, 4], dtype=np.int32),
        "assume_unique": True,
        "size": 3,
        "fill_value": 100,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Single-element arrays
    input_dict = {
        "ar1": np.array([42], dtype=np.int64),
        "ar2": np.array([24], dtype=np.int64),
        "assume_unique": True,
        "size": 1,
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.setdiff1d_3"] = setdiff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.setdiff1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.setdiff1d_3'.")


check_valid('jax.numpy.setdiff1d', generated_inputs['jax.numpy.setdiff1d_3'], lib="jax", suffix=3)
