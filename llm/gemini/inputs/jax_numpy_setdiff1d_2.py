
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def setdiff1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with some overlapping elements
    list_of_inputs.append({
        "ar1": np.array([1, 2, 3, 4], dtype=np.int32),
        "ar2": np.array([3, 4, 5, 6], dtype=np.int32),
        "assume_unique": False,
        "size": 2,
        "fill_value": 0.0
    })

    # Input 2: Floats, assuming uniqueness, size too large requiring padding
    list_of_inputs.append({
        "ar1": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        "ar2": np.array([3.5, 4.5, 5.5], dtype=np.float32),
        "assume_unique": True,
        "size": 3,
        "fill_value": -1.0
    })

    # Input 3: Negative and positive integers, size larger than diff output
    list_of_inputs.append({
        "ar1": np.array([-10, -20, -30, 0, 10, 20], dtype=np.int64),
        "ar2": np.array([-20, 10, 30], dtype=np.int64),
        "assume_unique": False,
        "size": 5,
        "fill_value": -999.0
    })

    # Input 4: Duplicated elements in ar1, truncation size 1
    list_of_inputs.append({
        "ar1": np.array([1, 1, 2, 2, 3, 3], dtype=np.int32),
        "ar2": np.array([1, 2], dtype=np.int32),
        "assume_unique": False,
        "size": 1,
        "fill_value": 0.0
    })

    # Input 5: Float64 inputs, no overlap, padded output
    list_of_inputs.append({
        "ar1": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "ar2": np.array([4.0, 5.0, 6.0], dtype=np.float64),
        "assume_unique": True,
        "size": 4,
        "fill_value": -99.9
    })

    # Input 6: Single-element arrays, complete overlap
    list_of_inputs.append({
        "ar1": np.array([100], dtype=np.int32),
        "ar2": np.array([100], dtype=np.int32),
        "assume_unique": False,
        "size": 1,
        "fill_value": 42.0
    })

    # Input 7: Empty ar1, requiring size padding
    list_of_inputs.append({
        "ar1": np.array([], dtype=np.float32),
        "ar2": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "assume_unique": True,
        "size": 2,
        "fill_value": 0.0
    })

    # Input 8: Unordered inputs, exact size fit
    list_of_inputs.append({
        "ar1": np.array([5, 4, 3, 2, 1], dtype=np.int32),
        "ar2": np.array([2, 3], dtype=np.int32),
        "assume_unique": False,
        "size": 3,
        "fill_value": 99.0
    })

    # Input 9: Fractional values, float64, size larger than possible output
    list_of_inputs.append({
        "ar1": np.array([-1.2, 3.4, -5.6, 7.8], dtype=np.float64),
        "ar2": np.array([3.4, 9.0], dtype=np.float64),
        "assume_unique": True,
        "size": 5,
        "fill_value": -1.0
    })

    # Input 10: Even and odd integers, subset comparison, exact size matches output
    list_of_inputs.append({
        "ar1": np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32),
        "ar2": np.array([20, 40, 60, 80], dtype=np.int32),
        "assume_unique": True,
        "size": 4,
        "fill_value": 0.0
    })

    return list_of_inputs

generated_inputs["jax.numpy.setdiff1d_2"] = setdiff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.setdiff1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.setdiff1d_2'.")


check_valid('jax.numpy.setdiff1d', generated_inputs['jax.numpy.setdiff1d_2'], lib="jax", suffix=2)
