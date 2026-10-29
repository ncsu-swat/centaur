
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def beta_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, standard interval [0, 1]
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "a": np.array(2.0, dtype=np.float32),
        "b": np.array(5.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with float64
    input_dict = {
        "x": np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float64),
        "a": np.array([1.5, 2.0, 0.5, 3.0], dtype=np.float64),
        "b": np.array([2.5, 1.0, 0.5, 1.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-zero location and scale
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.5, 2.0], dtype=np.float32),
        "a": np.array(1.0, dtype=np.float32),
        "b": np.array(1.0, dtype=np.float32),
        "loc": np.array(-2.0, dtype=np.float32),
        "scale": np.array(5.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays
    input_dict = {
        "x": np.array([[0.2, 0.3], [0.6, 0.8]], dtype=np.float32),
        "a": np.array([[2.0, 3.0], [1.0, 0.5]], dtype=np.float32),
        "b": np.array([[3.0, 2.0], [0.5, 1.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Values outside support (x < loc and x > loc + scale)
    input_dict = {
        "x": np.array([-0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32),
        "a": np.array(2.0, dtype=np.float32),
        "b": np.array(2.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting shapes (3, 1) and (1, 4)
    input_dict = {
        "x": np.array([[0.1], [0.5], [0.9]], dtype=np.float32),
        "a": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        "b": np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32),
        "loc": np.array([[0.0]], dtype=np.float32),
        "scale": np.array([[1.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor input
    input_dict = {
        "x": np.full((2, 3, 4), 0.4, dtype=np.float32),
        "a": np.full((2, 3, 4), 3.0, dtype=np.float32),
        "b": np.full((2, 3, 4), 4.0, dtype=np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
        "scale": np.ones((2, 3, 4), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small shape parameters (U-shaped distribution)
    input_dict = {
        "x": np.array([0.05, 0.1, 0.9, 0.95], dtype=np.float64),
        "a": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        "b": np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large shape parameters (bell-shaped distribution)
    input_dict = {
        "x": np.array([49.0, 50.0, 51.0], dtype=np.float32),
        "a": np.array([50.0, 50.0, 50.0], dtype=np.float32),
        "b": np.array([50.0, 50.0, 50.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([100.0, 100.0, 100.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative loc and fractional scale
    input_dict = {
        "x": np.array([-4.9, -4.5, -4.1], dtype=np.float64),
        "a": np.array([2.5, 2.5, 2.5], dtype=np.float64),
        "b": np.array([1.5, 1.5, 1.5], dtype=np.float64),
        "loc": np.array([-5.0, -5.0, -5.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.beta.cdf"] = beta_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.beta.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.beta.cdf'.")


check_valid('jax.scipy.stats.beta.cdf', generated_inputs['jax.scipy.stats.beta.cdf'], lib="jax", suffix=0)
