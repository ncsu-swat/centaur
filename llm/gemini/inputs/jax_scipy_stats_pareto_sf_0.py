
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pareto_sf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays
    input_dict = {
        "x": np.array(2.5, dtype=np.float32),
        "b": np.array(1.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with standard values
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "b": np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float64
    input_dict = {
        "x": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64),
        "b": np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64),
        "loc": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting with scalar/0D parameters
    input_dict = {
        "x": np.array([0.5, 1.5, 2.5, 5.0, 10.0], dtype=np.float32),
        "b": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative loc offset
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "b": np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float32),
        "loc": np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float32),
        "scale": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x below lower bound (x - loc < scale)
    input_dict = {
        "x": np.array([0.1, 0.5, 0.9], dtype=np.float32),
        "b": np.array([2.5, 2.5, 2.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array inputs
    input_dict = {
        "x": np.full((2, 3, 4), 5.0, dtype=np.float32),
        "b": np.full((2, 3, 4), 1.2, dtype=np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
        "scale": np.ones((2, 3, 4), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting 1D parameters across 2D x
    input_dict = {
        "x": np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float64),
        "b": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "loc": np.array([0.0, 1.0, 2.0], dtype=np.float64),
        "scale": np.array([1.0, 0.5, 2.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-trivial scale and location with large values
    input_dict = {
        "x": np.array([100.0, 200.0, 500.0], dtype=np.float32),
        "b": np.array([0.8, 1.2, 2.1], dtype=np.float32),
        "loc": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "scale": np.array([5.0, 10.0, 15.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional broadcasting (column vector and row vector)
    input_dict = {
        "x": np.array([[2.0], [4.0], [6.0]], dtype=np.float32),
        "b": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "loc": np.array([[0.0]], dtype=np.float32),
        "scale": np.array([[1.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Small shape parameter b
    input_dict = {
        "x": np.array([10.0, 100.0, 1000.0], dtype=np.float64),
        "b": np.array([0.01, 0.05, 0.1], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.pareto.sf"] = pareto_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.sf'.")


check_valid('jax.scipy.stats.pareto.sf', generated_inputs['jax.scipy.stats.pareto.sf'], lib="jax", suffix=0)
