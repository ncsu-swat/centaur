
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def geom_logpmf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard case
    input_dict = {
        "k": np.array(2, dtype=np.int32),
        "p": np.array(0.5, dtype=np.float32),
        "loc": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with matching dimensions
    input_dict = {
        "k": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "p": np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32),
        "loc": np.array([0, 0, 0, 0, 0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float k values, float64 precision
    input_dict = {
        "k": np.array([1.0, 3.0, 5.0], dtype=np.float64),
        "p": np.array([0.3, 0.5, 0.7], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Non-zero loc parameter
    input_dict = {
        "k": np.array([5, 6, 7], dtype=np.int32),
        "p": np.array([0.4, 0.4, 0.4], dtype=np.float32),
        "loc": np.array([3, 3, 3], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting 1D to 2D
    input_dict = {
        "k": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "p": np.array([0.25], dtype=np.float32),
        "loc": np.array([0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D arrays with distinct p values per row
    input_dict = {
        "k": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "p": np.array([[0.1], [0.9]], dtype=np.float32),
        "loc": np.array([[0], [0]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array inputs
    input_dict = {
        "k": np.ones((2, 3, 4), dtype=np.int32) * 2,
        "p": np.full((2, 3, 4), 0.6, dtype=np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative loc values shifting support
    input_dict = {
        "k": np.array([-1, 0, 1, 2], dtype=np.int32),
        "p": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "loc": np.array([-3, -3, -3, -3], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Extreme p values (near boundary 0 and 1)
    input_dict = {
        "k": np.array([1, 2, 10], dtype=np.int64),
        "p": np.array([1e-4, 0.9999, 0.5], dtype=np.float64),
        "loc": np.array([0, 0, 0], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting with column and row vectors
    input_dict = {
        "k": np.arange(1, 5, dtype=np.int32).reshape(4, 1),
        "p": np.array([0.1, 0.3, 0.7], dtype=np.float32).reshape(1, 3),
        "loc": np.array([0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Values out of support (k < 1 + loc) where logpmf produces -inf
    input_dict = {
        "k": np.array([0, -1, 1], dtype=np.int32),
        "p": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "loc": np.array([0, 0, 0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.geom.logpmf"] = geom_logpmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.geom.logpmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.geom.logpmf'.")


check_valid('jax.scipy.stats.geom.logpmf', generated_inputs['jax.scipy.stats.geom.logpmf'], lib="jax", suffix=0)
