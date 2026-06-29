
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def chi2_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Basic scalar-like arrays
    input_dict = {
        "x": np.array(2.0, dtype=np.float32),
        "df": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with standard values
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "df": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays
    input_dict = {
        "x": np.ones((3, 3), dtype=np.float32) * 5.0,
        "df": np.ones((3, 3), dtype=np.float32) * 2.0,
        "loc": np.zeros((3, 3), dtype=np.float32),
        "scale": np.ones((3, 3), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 precision and customized location/scale
    input_dict = {
        "x": np.array([2.5, 3.5, 4.5, 5.5], dtype=np.float64),
        "df": np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float64),
        "loc": np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64),
        "scale": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative value test for x
    input_dict = {
        "x": np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        "df": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays
    input_dict = {
        "x": np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2),
        "df": np.ones((2, 2, 2), dtype=np.float32) * 4.0,
        "loc": np.zeros((2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2), dtype=np.float32) * 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting with 1D and 0D arrays
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "df": np.array(2.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-dimensional broadcasting
    input_dict = {
        "x": np.ones((2, 3), dtype=np.float32) * 4.0,
        "df": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        "loc": np.array([[0.5], [1.0]], dtype=np.float32),
        "scale": np.array(0.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values for x and df
    input_dict = {
        "x": np.array([100.0, 1000.0], dtype=np.float32),
        "df": np.array([50.0, 100.0], dtype=np.float32),
        "loc": np.array([10.0, 20.0], dtype=np.float32),
        "scale": np.array([5.0, 10.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Fractional df (less than 1) and small scale
    input_dict = {
        "x": np.array([0.1, 0.5, 1.0], dtype=np.float32),
        "df": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([0.1, 0.5, 1.0], dtype=np.float32)
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
