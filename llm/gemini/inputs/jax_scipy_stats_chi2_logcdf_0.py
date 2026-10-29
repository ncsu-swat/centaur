
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def chi2_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    input_dict = {
        "x": np.array(2.5, dtype=np.float32),
        "df": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, identical shapes, float64
    input_dict = {
        "x": np.array([0.5, 1.5, 3.0, 5.0], dtype=np.float64),
        "df": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-zero loc and non-unit scale
    input_dict = {
        "x": np.array([2.0, 4.0, 6.0], dtype=np.float32),
        "df": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array broadcast with 1D parameters
    input_dict = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "df": np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Values of x below loc (support region edge cases)
    input_dict = {
        "x": np.array([-1.0, 0.0, 0.5], dtype=np.float32),
        "df": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large degrees of freedom
    input_dict = {
        "x": np.array([50.0, 100.0, 150.0], dtype=np.float64),
        "df": np.array([100.0, 100.0, 100.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Fractional degrees of freedom
    input_dict = {
        "x": np.array([0.1, 0.5, 1.2], dtype=np.float32),
        "df": np.array([0.5, 1.5, 2.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([0.5, 1.5, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensors
    input_dict = {
        "x": np.ones((2, 3, 4), dtype=np.float32) * 3.0,
        "df": np.ones((2, 3, 4), dtype=np.float32) * 5.0,
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
        "scale": np.ones((2, 3, 4), dtype=np.float32) * 2.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative loc offset
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "df": np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32),
        "loc": np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float32),
        "scale": np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting dimensions (2D with 1D and scalar tensors)
    input_dict = {
        "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        "df": np.array([[2.0], [4.0]], dtype=np.float64),
        "loc": np.array([0.0, 0.5, 1.0], dtype=np.float64),
        "scale": np.array([[1.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.chi2.logcdf"] = chi2_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.chi2.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.chi2.logcdf'.")


check_valid('jax.scipy.stats.chi2.logcdf', generated_inputs['jax.scipy.stats.chi2.logcdf'], lib="jax", suffix=0)
