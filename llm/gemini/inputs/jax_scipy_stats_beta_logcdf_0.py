
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def beta_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32, standard domain
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "a": np.array(2.0, dtype=np.float32),
        "b": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, float64, standard domain
    input_dict = {
        "x": np.array([0.1, 0.4, 0.8], dtype=np.float64),
        "a": np.array([0.5, 1.0, 2.0], dtype=np.float64),
        "b": np.array([2.0, 1.5, 0.5], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, non-standard loc and scale
    input_dict = {
        "x": np.array([[1.5, 2.0], [2.5, 2.8]], dtype=np.float32),
        "a": np.array([[2.0, 3.0], [1.0, 2.5]], dtype=np.float32),
        "b": np.array([[3.0, 2.0], [4.0, 1.5]], dtype=np.float32),
        "loc": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "scale": np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting shapes (1D with 2D)
    input_dict = {
        "x": np.array([[0.2, 0.5, 0.7]], dtype=np.float32),
        "a": np.array([[1.0], [2.0]], dtype=np.float32),
        "b": np.array([[3.0], [1.5]], dtype=np.float32),
        "loc": np.array([[0.0]], dtype=np.float32),
        "scale": np.array([[1.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensors, with negative loc
    input_dict = {
        "x": np.array([[[-0.5, 0.0], [0.5, 0.8]]], dtype=np.float64),
        "a": np.array([[[1.5, 2.0], [0.8, 1.2]]], dtype=np.float64),
        "b": np.array([[[2.5, 1.0], [3.0, 2.0]]], dtype=np.float64),
        "loc": np.array([[[-1.0, -1.0], [-1.0, -1.0]]], dtype=np.float64),
        "scale": np.array([[[2.0, 2.0], [2.0, 2.0]]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x below support (x <= loc)
    input_dict = {
        "x": np.array([-0.2, -0.5], dtype=np.float32),
        "a": np.array([2.0, 3.0], dtype=np.float32),
        "b": np.array([2.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x above support (x >= loc + scale)
    input_dict = {
        "x": np.array([1.2, 1.5, 2.0], dtype=np.float64),
        "a": np.array([2.0, 1.0, 0.5], dtype=np.float64),
        "b": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensors
    input_dict = {
        "x": np.full((1, 2, 2, 1), 0.3, dtype=np.float32),
        "a": np.full((1, 2, 2, 1), 1.5, dtype=np.float32),
        "b": np.full((1, 2, 2, 1), 2.5, dtype=np.float32),
        "loc": np.full((1, 2, 2, 1), 0.0, dtype=np.float32),
        "scale": np.full((1, 2, 2, 1), 1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large parameter values
    input_dict = {
        "x": np.array([0.45, 0.5, 0.55], dtype=np.float64),
        "a": np.array([50.0, 100.0, 200.0], dtype=np.float64),
        "b": np.array([50.0, 100.0, 200.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed broadcasting shapes with varying scale
    input_dict = {
        "x": np.array([1.0, 3.0, 5.0], dtype=np.float32),
        "a": np.array([[0.5], [2.0]], dtype=np.float32),
        "b": np.array([[1.5], [0.5]], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([2.0, 5.0, 10.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.beta.logcdf"] = beta_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.beta.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.beta.logcdf'.")


check_valid('jax.scipy.stats.beta.logcdf', generated_inputs['jax.scipy.stats.beta.logcdf'], lib="jax", suffix=0)
