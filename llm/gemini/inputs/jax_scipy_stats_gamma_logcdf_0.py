
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gamma_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar tensors (float32)
    input_dict = {
        "x": np.array(2.5, dtype=np.float32),
        "a": np.array(1.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, float64
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "a": np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float32
    input_dict = {
        "x": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        "a": np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        "loc": np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32),
        "scale": np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting 1D with scalar-like arrays
    input_dict = {
        "x": np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        "a": np.array([2.0], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays
    input_dict = {
        "x": np.full((2, 3, 4), 3.0, dtype=np.float32),
        "a": np.full((2, 3, 4), 1.2, dtype=np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
        "scale": np.full((2, 3, 4), 0.5, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting multidimensional shapes (e.g., (3, 1) and (1, 4))
    input_dict = {
        "x": np.array([[1.0], [2.0], [3.0]], dtype=np.float64),
        "a": np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float64),
        "loc": np.array([[0.0]], dtype=np.float64),
        "scale": np.array([[1.0, 2.0, 0.5, 1.5]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative loc parameter
    input_dict = {
        "x": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "a": np.array([2.5, 3.0, 3.5], dtype=np.float32),
        "loc": np.array([-2.0, -1.0, -0.5], dtype=np.float32),
        "scale": np.array([1.0, 1.5, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small shape parameter (a < 1)
    input_dict = {
        "x": np.array([0.1, 0.5, 1.0], dtype=np.float32),
        "a": np.array([0.1, 0.2, 0.05], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([0.5, 1.0, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values for x and large scale
    input_dict = {
        "x": np.array([100.0, 200.0, 500.0], dtype=np.float64),
        "a": np.array([5.0, 10.0, 20.0], dtype=np.float64),
        "loc": np.array([10.0, 20.0, 50.0], dtype=np.float64),
        "scale": np.array([10.0, 20.0, 25.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: x values around and below loc (left tail)
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "a": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D tensor input
    input_dict = {
        "x": np.ones((2, 2, 2, 2), dtype=np.float32) * 3.0,
        "a": np.ones((2, 2, 2, 2), dtype=np.float32) * 2.0,
        "loc": np.zeros((2, 2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2, 2), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gamma.logcdf"] = gamma_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gamma.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gamma.logcdf'.")


check_valid('jax.scipy.stats.gamma.logcdf', generated_inputs['jax.scipy.stats.gamma.logcdf'], lib="jax", suffix=0)
