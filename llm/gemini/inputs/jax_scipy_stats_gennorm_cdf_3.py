
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gennorm_cdf_inputs():
    list_of_inputs = []

    # Input 1: Basic Python integers
    input_dict = {
        "x": 0,
        "beta": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative x and positive beta as Python integers
    input_dict = {
        "x": -5,
        "beta": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D NumPy arrays of int32
    input_dict = {
        "x": np.array([1, 2, 3], dtype=np.int32),
        "beta": np.array([1, 2, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D NumPy arrays of int64 with negative values in x
    input_dict = {
        "x": np.array([-2, 0, 2], dtype=np.int64),
        "beta": np.array([2, 2, 2], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D NumPy arrays of int32
    input_dict = {
        "x": np.array([[1, -1], [2, -2]], dtype=np.int32),
        "beta": np.array([[1, 2], [3, 4]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1-element NumPy arrays
    input_dict = {
        "x": np.array([5], dtype=np.int32),
        "beta": np.array([2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array for x and scalar for beta (broadcasting)
    input_dict = {
        "x": np.array([[-3, 4], [0, 1]], dtype=np.int64),
        "beta": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar x and 1D array for beta
    input_dict = {
        "x": 1,
        "beta": np.array([1, 2, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D NumPy arrays of int32
    input_dict = {
        "x": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "beta": np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 1D array of int32 with negative and positive values
    input_dict = {
        "x": np.array([-10, -5, 0, 5, 10], dtype=np.int32),
        "beta": np.array([1, 1, 1, 1, 1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gennorm.cdf_3"] = gennorm_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.cdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.cdf_3'.")


check_valid('jax.scipy.stats.gennorm.cdf', generated_inputs['jax.scipy.stats.gennorm.cdf_3'], lib="jax", suffix=3)
