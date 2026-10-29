
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nbinom_logpmf_inputs():
    list_of_inputs = []

    # Input 1: Basic 0D scalar arrays
    input_dict = {
        "k": np.array(5, dtype=np.int32),
        "n": np.array(10, dtype=np.int32),
        "p": np.array(0.5, dtype=np.float32),
        "loc": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays of matching shape
    input_dict = {
        "k": np.array([0, 1, 2, 5, 10], dtype=np.int32),
        "n": np.array([5, 5, 5, 5, 5], dtype=np.float32),
        "p": np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        "loc": np.array([0, 0, 0, 0, 0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-zero loc parameter
    input_dict = {
        "k": np.array([2, 3, 4, 5], dtype=np.int64),
        "n": np.array([4, 4, 4, 4], dtype=np.int64),
        "p": np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float64),
        "loc": np.array([2, 2, 2, 2], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting 1D k with 0D parameters
    input_dict = {
        "k": np.arange(10, dtype=np.float32),
        "n": np.array(3.5, dtype=np.float32),
        "p": np.array(0.6, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays
    input_dict = {
        "k": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "n": np.array([[5, 6], [7, 8]], dtype=np.float32),
        "p": np.array([[0.2, 0.5], [0.7, 0.9]], dtype=np.float32),
        "loc": np.array([[0, 1], [0, 2]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting (2, 1) and (1, 3) -> (2, 3)
    input_dict = {
        "k": np.array([[1], [3]], dtype=np.int32),
        "n": np.array([[2, 4, 6]], dtype=np.float32),
        "p": np.array([[0.25]], dtype=np.float32),
        "loc": np.array([[0]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 high precision
    input_dict = {
        "k": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        "n": np.array([15.5, 25.5, 35.5], dtype=np.float64),
        "p": np.array([0.45, 0.55, 0.65], dtype=np.float64),
        "loc": np.array([0.0, 5.0, 10.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays
    input_dict = {
        "k": np.ones((2, 3, 4), dtype=np.int32) * 5,
        "n": np.ones((2, 3, 4), dtype=np.float32) * 10.0,
        "p": np.full((2, 3, 4), 0.35, dtype=np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative loc (shifting support into negative territory)
    input_dict = {
        "k": np.array([-2, -1, 0, 1, 2], dtype=np.int32),
        "n": np.array(5.0, dtype=np.float32),
        "p": np.array(0.5, dtype=np.float32),
        "loc": np.array(-3, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-integer real-valued n (shape parameter)
    input_dict = {
        "k": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "n": np.array([1.2, 2.7, 0.5], dtype=np.float32),
        "p": np.array([0.8, 0.4, 0.1], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Edge cases of probability p near boundaries
    input_dict = {
        "k": np.array([0, 1, 2], dtype=np.int32),
        "n": np.array([1, 1, 1], dtype=np.int32),
        "p": np.array([1e-4, 0.9999, 1.0], dtype=np.float64),
        "loc": np.array([0, 0, 0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.nbinom.logpmf"] = nbinom_logpmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.nbinom.logpmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.nbinom.logpmf'.")


check_valid('jax.scipy.stats.nbinom.logpmf', generated_inputs['jax.scipy.stats.nbinom.logpmf'], lib="jax", suffix=0)
