
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pareto_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (standard float32)
    input_dict = {
        "x": np.array(2.5, dtype=np.float32),
        "b": np.array(1.5, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with matching shapes
    input_dict = {
        "x": np.array([1.5, 2.0, 3.5, 10.0], dtype=np.float32),
        "b": np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 precision with non-zero loc and non-unit scale
    input_dict = {
        "x": np.array([5.0, 7.5, 12.0], dtype=np.float64),
        "b": np.array([2.5, 1.8, 3.2], dtype=np.float64),
        "loc": np.array([1.0, 2.0, 0.5], dtype=np.float64),
        "scale": np.array([2.0, 1.5, 3.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays
    input_dict = {
        "x": np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        "b": np.array([[1.0, 2.0], [1.5, 0.5]], dtype=np.float32),
        "loc": np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting 1D parameters with 2D x
    input_dict = {
        "x": np.ones((3, 4), dtype=np.float32) * 5.0,
        "b": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Values of x below support (loc + scale), expecting logcdf to evaluate to -inf
    input_dict = {
        "x": np.array([0.0, 0.5, -1.0], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensors
    input_dict = {
        "x": np.full((2, 3, 2), 4.0, dtype=np.float32),
        "b": np.full((2, 3, 2), 2.5, dtype=np.float32),
        "loc": np.zeros((2, 3, 2), dtype=np.float32),
        "scale": np.ones((2, 3, 2), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar-shaped 1D arrays
    input_dict = {
        "x": np.array([100.0], dtype=np.float64),
        "b": np.array([5.0], dtype=np.float64),
        "loc": np.array([-10.0], dtype=np.float64),
        "scale": np.array([0.1], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values of x testing asymptotic logcdf behavior (close to 0)
    input_dict = {
        "x": np.array([1e3, 1e5, 1e8], dtype=np.float32),
        "b": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative loc with values of x within support
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "b": np.array([1.2, 1.2, 1.2, 1.2], dtype=np.float32),
        "loc": np.array([-3.0, -3.0, -3.0, -3.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.pareto.logcdf"] = pareto_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.logcdf'.")


check_valid('jax.scipy.stats.pareto.logcdf', generated_inputs['jax.scipy.stats.pareto.logcdf'], lib="jax", suffix=0)
