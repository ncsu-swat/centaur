
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def truncnorm_logsf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard interval [-1, 1]
    input_dict = {
        "x": np.array(0.0, dtype=np.float32),
        "a": np.array(-1.0, dtype=np.float32),
        "b": np.array(1.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, float32, simple standard normal truncated to [0, 2]
    input_dict = {
        "x": np.array([0.2, 0.5, 1.0, 1.8], dtype=np.float32),
        "a": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays, float64, non-zero loc and non-unit scale
    input_dict = {
        "x": np.array([3.5, 4.0, 5.2], dtype=np.float64),
        "a": np.array([-1.5, -1.5, -1.5], dtype=np.float64),
        "b": np.array([2.5, 2.5, 2.5], dtype=np.float64),
        "loc": np.array([4.0, 4.0, 4.0], dtype=np.float64),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, both a and b strictly positive
    input_dict = {
        "x": np.array([[1.2, 1.5], [2.0, 2.8]], dtype=np.float32),
        "a": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "b": np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays, both a and b strictly negative
    input_dict = {
        "x": np.array([[-2.5, -2.0], [-1.8, -1.2]], dtype=np.float32),
        "a": np.array([[-3.0, -3.0], [-3.0, -3.0]], dtype=np.float32),
        "b": np.array([[-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting shapes (x is (3, 1), bounds are (1, 4))
    input_dict = {
        "x": np.array([[-0.5], [0.0], [0.5]], dtype=np.float32),
        "a": np.array([[-2.0, -1.5, -1.0, -0.8]], dtype=np.float32),
        "b": np.array([[0.8, 1.0, 1.5, 2.0]], dtype=np.float32),
        "loc": np.array([[0.0]], dtype=np.float32),
        "scale": np.array([[1.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays, float64
    input_dict = {
        "x": np.full((2, 2, 2), 0.5, dtype=np.float64),
        "a": np.full((2, 2, 2), -2.0, dtype=np.float64),
        "b": np.full((2, 2, 2), 2.0, dtype=np.float64),
        "loc": np.full((2, 2, 2), 0.0, dtype=np.float64),
        "scale": np.full((2, 2, 2), 0.5, dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x values outside the support range (evaluating tails)
    input_dict = {
        "x": np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32),
        "a": np.array([-2.0, -2.0, -2.0, -2.0, -2.0], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Varying bounds per element in 1D array
    input_dict = {
        "x": np.array([0.1, 1.2, -0.3], dtype=np.float32),
        "a": np.array([-1.0, 0.5, -2.0], dtype=np.float32),
        "b": np.array([1.0, 2.5, -0.1], dtype=np.float32),
        "loc": np.array([0.0, 1.0, -1.0], dtype=np.float32),
        "scale": np.array([0.5, 1.5, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional array with varying locations and scales
    input_dict = {
        "x": np.array([[10.0, 12.0], [8.0, 15.0]], dtype=np.float64),
        "a": np.array([[-2.0, -1.0], [-3.0, 0.0]], dtype=np.float64),
        "b": np.array([[1.0, 2.0], [0.5, 3.0]], dtype=np.float64),
        "loc": np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float64),
        "scale": np.array([[3.0, 2.0], [1.5, 4.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.truncnorm.logsf"] = truncnorm_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.truncnorm.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.truncnorm.logsf'.")


check_valid('jax.scipy.stats.truncnorm.logsf', generated_inputs['jax.scipy.stats.truncnorm.logsf'], lib="jax", suffix=0)
