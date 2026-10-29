
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def chi2_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, float32
    input_dict = {
        "x": np.array(2.5, dtype=np.float32),
        "df": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, float64
    input_dict = {
        "x": np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float64),
        "df": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x values (below loc)
    input_dict = {
        "x": np.array([-2.0, -0.5, 0.0], dtype=np.float32),
        "df": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Non-zero loc and non-unit scale
    input_dict = {
        "x": np.array([3.0, 5.0, 8.0], dtype=np.float32),
        "df": np.array([4.0, 4.0, 4.0], dtype=np.float32),
        "loc": np.array([1.0, 2.0, 0.5], dtype=np.float32),
        "scale": np.array([2.0, 1.5, 0.8], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Fractional degrees of freedom
    input_dict = {
        "x": np.array([0.1, 0.5, 1.2], dtype=np.float64),
        "df": np.array([0.5, 1.5, 2.5], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D arrays with broadcasting
    input_dict = {
        "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "df": np.array([[2.0], [5.0]], dtype=np.float32),
        "loc": np.array([[0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 2.0, 1.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large degrees of freedom
    input_dict = {
        "x": np.array([90.0, 100.0, 110.0], dtype=np.float64),
        "df": np.array([100.0, 100.0, 100.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays
    input_dict = {
        "x": np.ones((2, 3, 4), dtype=np.float32) * 3.0,
        "df": np.ones((2, 3, 4), dtype=np.float32) * 4.0,
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
        "scale": np.ones((2, 3, 4), dtype=np.float32) * 2.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Integer values cast to float arrays
    input_dict = {
        "x": np.array([1, 2, 3, 4], dtype=np.int32).astype(np.float32),
        "df": np.array([2, 2, 2, 2], dtype=np.int32).astype(np.float32),
        "loc": np.array([0, 0, 0, 0], dtype=np.int32).astype(np.float32),
        "scale": np.array([1, 1, 1, 1], dtype=np.int32).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D square matrix with varied scale and loc
    input_dict = {
        "x": np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64),
        "df": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "loc": np.array([[-0.5, 0.0], [0.5, 1.0]], dtype=np.float64),
        "scale": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.chi2.cdf"] = chi2_cdf_inputs()

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
