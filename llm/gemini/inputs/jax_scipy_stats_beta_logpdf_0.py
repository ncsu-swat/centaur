
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def beta_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "a": np.array(2.0, dtype=np.float32),
        "b": np.array(2.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, same shapes, float64
    input_dict = {
        "x": np.array([0.1, 0.5, 0.9], dtype=np.float64),
        "a": np.array([0.5, 1.0, 2.0], dtype=np.float64),
        "b": np.array([0.5, 1.0, 2.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-standard loc and scale, 1D
    input_dict = {
        "x": np.array([1.5, 2.0, 2.5], dtype=np.float32),
        "a": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        "b": np.array([3.0, 2.0, 1.0], dtype=np.float32),
        "loc": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative loc, float32
    input_dict = {
        "x": np.array([-0.5, 0.0, 0.5], dtype=np.float32),
        "a": np.array([1.5, 1.5, 1.5], dtype=np.float32),
        "b": np.array([2.5, 2.5, 2.5], dtype=np.float32),
        "loc": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays
    input_dict = {
        "x": np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
        "a": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "b": np.array([[4.0, 3.0], [2.0, 1.0]], dtype=np.float32),
        "loc": np.zeros((2, 2), dtype=np.float32),
        "scale": np.ones((2, 2), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting shapes (2D x with 1D params)
    input_dict = {
        "x": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "b": np.array([[2.0], [3.0]], dtype=np.float64),
        "loc": np.array([0.0], dtype=np.float64),
        "scale": np.array([1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensors
    input_dict = {
        "x": np.full((2, 3, 4), 0.3, dtype=np.float32),
        "a": np.full((2, 3, 4), 2.5, dtype=np.float32),
        "b": np.full((2, 3, 4), 1.5, dtype=np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
        "scale": np.ones((2, 3, 4), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D parameters with multi-dimensional x
    input_dict = {
        "x": np.linspace(0.1, 0.9, 10, dtype=np.float32).reshape(5, 2),
        "a": np.array(5.0, dtype=np.float32),
        "b": np.array(2.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Boundary conditions (support boundary values)
    input_dict = {
        "x": np.array([0.0, 0.5, 1.0], dtype=np.float64),
        "a": np.array([2.0, 2.0, 2.0], dtype=np.float64),
        "b": np.array([2.0, 2.0, 2.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Asymmetric shape parameters with large scale
    input_dict = {
        "x": np.array([10.0, 25.0, 50.0, 90.0], dtype=np.float32),
        "a": np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32),
        "b": np.array([5.0, 2.0, 1.0, 0.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.beta.logpdf"] = beta_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.beta.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.beta.logpdf'.")


check_valid('jax.scipy.stats.beta.logpdf', generated_inputs['jax.scipy.stats.beta.logpdf'], lib="jax", suffix=0)
