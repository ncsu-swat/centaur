
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def truncnorm_cdf_inputs():
    list_of_inputs = []

    # Input 1: Scalar-like values (0D arrays)
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "a": np.array(-1.0, dtype=np.float32),
        "b": np.array(2.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, standard normal range
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "a": np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 1D arrays, non-zero mean and non-unit scale
    input_dict = {
        "x": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "a": np.array([-3.0, -2.0, -1.0], dtype=np.float64),
        "b": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "loc": np.array([0.5, 0.5, 0.5], dtype=np.float64),
        "scale": np.array([0.5, 1.5, 2.5], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays
    input_dict = {
        "x": np.random.uniform(-1, 1, (3, 3)).astype(np.float32),
        "a": np.full((3, 3), -2.0, dtype=np.float32),
        "b": np.full((3, 3), 2.0, dtype=np.float32),
        "loc": np.zeros((3, 3), dtype=np.float32),
        "scale": np.ones((3, 3), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative loc, scale > 1, truncation bounds both negative (a < b < 0)
    input_dict = {
        "x": np.array([-5.0, -2.0, 0.0, 3.0], dtype=np.float32),
        "a": np.array([-3.0, -3.0, -3.0, -3.0], dtype=np.float32),
        "b": np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
        "loc": np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
        "scale": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays
    input_dict = {
        "x": np.zeros((2, 2, 2), dtype=np.float32),
        "a": np.ones((2, 2, 2), dtype=np.float32) * -1.5,
        "b": np.ones((2, 2, 2), dtype=np.float32) * 1.5,
        "loc": np.zeros((2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2), dtype=np.float32) * 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x below the lower truncation bound (CDF should be 0)
    input_dict = {
        "x": np.array([-10.0], dtype=np.float32),
        "a": np.array([-2.0], dtype=np.float32),
        "b": np.array([2.0], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x above the upper truncation bound (CDF should be 1)
    input_dict = {
        "x": np.array([10.0], dtype=np.float32),
        "a": np.array([-2.0], dtype=np.float32),
        "b": np.array([2.0], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D arrays of float32
    input_dict = {
        "x": np.random.uniform(-1, 1, (2, 2, 2, 2)).astype(np.float32),
        "a": np.full((2, 2, 2, 2), -5.0, dtype=np.float32),
        "b": np.full((2, 2, 2, 2), 5.0, dtype=np.float32),
        "loc": np.full((2, 2, 2, 2), 0.1, dtype=np.float32),
        "scale": np.full((2, 2, 2, 2), 1.2, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Varying truncation bounds per element
    input_dict = {
        "x": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "a": np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        "b": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.truncnorm.cdf"] = truncnorm_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.truncnorm.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.truncnorm.cdf'.")


check_valid('jax.scipy.stats.truncnorm.cdf', generated_inputs['jax.scipy.stats.truncnorm.cdf'], lib="jax", suffix=0)
