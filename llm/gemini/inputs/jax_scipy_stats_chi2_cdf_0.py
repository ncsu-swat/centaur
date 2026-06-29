
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_chi2_cdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays, default-like loc and scale
    input_dict = {
        "x": np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32),
        "df": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays with some negative x values
    input_dict = {
        "x": np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32),
        "df": np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float64 arrays
    input_dict = {
        "x": np.array([0.1, 0.2, 0.5, 1.5], dtype=np.float64),
        "df": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64),
        "loc": np.array([0.1, -0.1, 0.0, 0.2], dtype=np.float64),
        "scale": np.array([0.5, 1.5, 2.0, 1.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D arrays (scalars as tensors)
    input_dict = {
        "x": np.array(2.5, dtype=np.float32),
        "df": np.array(4.0, dtype=np.float32),
        "loc": np.array(0.5, dtype=np.float32),
        "scale": np.array(1.2, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcastable dimensions
    input_dict = {
        "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "df": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        "loc": np.array([[0.5], [1.0]], dtype=np.float32),
        "scale": np.array([1.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensors
    input_dict = {
        "x": np.ones((2, 2, 2), dtype=np.float32) * 5.0,
        "df": np.ones((2, 2, 2), dtype=np.float32) * 3.0,
        "loc": np.zeros((2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2), dtype=np.float32) * 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large scale and negative loc
    input_dict = {
        "x": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "df": np.array([5.0, 5.0, 5.0], dtype=np.float32),
        "loc": np.array([-10.0, -10.0, -10.0], dtype=np.float32),
        "scale": np.array([10.0, 10.0, 10.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small df (fractional degrees of freedom)
    input_dict = {
        "x": np.array([0.01, 0.05, 0.1], dtype=np.float32),
        "df": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([0.1, 0.1, 0.1], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values, float64
    input_dict = {
        "x": np.array([100.0, 500.0, 1000.0], dtype=np.float64),
        "df": np.array([50.0, 150.0, 300.0], dtype=np.float64),
        "loc": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        "scale": np.array([2.0, 3.0, 4.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed x relative to loc
    input_dict = {
        "x": np.array([-5.0, -1.0, 0.0, 2.0, 10.0], dtype=np.float32),
        "df": np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.chi2.cdf"] = jax_scipy_stats_chi2_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.chi2.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.chi2.cdf'.")


check_valid('jax.scipy.stats.chi2.cdf', generated_inputs['jax.scipy.stats.chi2.cdf'], lib="jax", suffix=0)
