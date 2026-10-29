
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def truncnorm_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32, standard normal truncation bounds
    input_dict = {
        "x": np.array(0.0, dtype=np.float32),
        "a": np.array(-2.0, dtype=np.float32),
        "b": np.array(2.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, float64, non-standard loc and scale
    input_dict = {
        "x": np.array([1.0, 2.5, 3.0], dtype=np.float64),
        "a": np.array([-1.0, -1.0, -1.0], dtype=np.float64),
        "b": np.array([2.0, 2.0, 2.0], dtype=np.float64),
        "loc": np.array([2.0, 2.0, 2.0], dtype=np.float64),
        "scale": np.array([1.5, 1.5, 1.5], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float32
    input_dict = {
        "x": np.array([[-0.5, 0.5], [1.0, 1.5]], dtype=np.float32),
        "a": np.array([[-2.0, -2.0], [-1.0, -1.0]], dtype=np.float32),
        "b": np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.0], [0.5, 0.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting shapes (1D with scalar-like 1D params)
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "a": np.array([-2.5], dtype=np.float32),
        "b": np.array([2.5], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x outside truncation bounds (x <= loc + a * scale, x >= loc + b * scale)
    input_dict = {
        "x": np.array([-5.0, 0.0, 5.0], dtype=np.float64),
        "a": np.array([-1.5, -1.5, -1.5], dtype=np.float64),
        "b": np.array([1.5, 1.5, 1.5], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensors, float32
    input_dict = {
        "x": np.zeros((2, 3, 4), dtype=np.float32),
        "a": np.full((2, 3, 4), -3.0, dtype=np.float32),
        "b": np.full((2, 3, 4), 3.0, dtype=np.float32),
        "loc": np.ones((2, 3, 4), dtype=np.float32),
        "scale": np.full((2, 3, 4), 2.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: One-sided lower truncated normal (b is very large)
    input_dict = {
        "x": np.array([0.5, 1.5, 2.5, 5.0], dtype=np.float64),
        "a": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "b": np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: One-sided upper truncated normal (a is very negative)
    input_dict = {
        "x": np.array([-5.0, -2.5, -1.0, 0.0], dtype=np.float32),
        "a": np.array([-10.0, -10.0, -10.0, -10.0], dtype=np.float32),
        "b": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multidimensional broadcast with negative loc
    input_dict = {
        "x": np.array([[-2.0], [0.0], [2.0]], dtype=np.float64),
        "a": np.array([-3.0, -2.0], dtype=np.float64),
        "b": np.array([2.0, 3.0], dtype=np.float64),
        "loc": np.array([-1.0], dtype=np.float64),
        "scale": np.array([0.5], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Narrow interval truncation bounds
    input_dict = {
        "x": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "a": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "b": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.truncnorm.logcdf"] = truncnorm_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.truncnorm.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.truncnorm.logcdf'.")


check_valid('jax.scipy.stats.truncnorm.logcdf', generated_inputs['jax.scipy.stats.truncnorm.logcdf'], lib="jax", suffix=0)
