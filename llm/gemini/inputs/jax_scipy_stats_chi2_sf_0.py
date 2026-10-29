
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def chi2_sf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (float32)
    input_dict = {
        "x": np.array(2.5, dtype=np.float32),
        "df": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays of standard values
    input_dict = {
        "x": np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float64),
        "df": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-zero loc and non-unit scale with broadcasting (scalar loc/scale)
    input_dict = {
        "x": np.array([3.0, 5.0, 7.0], dtype=np.float32),
        "df": np.array([2.0, 4.0, 6.0], dtype=np.float32),
        "loc": np.array(1.0, dtype=np.float32),
        "scale": np.array(2.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays
    input_dict = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "df": np.array([[2.0, 2.0], [5.0, 5.0]], dtype=np.float64),
        "loc": np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64),
        "scale": np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x values less than loc (yielding sf = 1.0), including negative x
    input_dict = {
        "x": np.array([-5.0, -1.0, 0.0, 0.5], dtype=np.float32),
        "df": np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "loc": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large degrees of freedom and large x values
    input_dict = {
        "x": np.array([50.0, 100.0, 150.0], dtype=np.float64),
        "df": np.array([100.0, 100.0, 100.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor inputs
    input_dict = {
        "x": np.ones((2, 3, 4), dtype=np.float32) * 4.0,
        "df": np.ones((2, 3, 4), dtype=np.float32) * 2.0,
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
        "scale": np.ones((2, 3, 4), dtype=np.float32) * 0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Fractional degrees of freedom
    input_dict = {
        "x": np.array([0.1, 0.5, 1.5, 2.5], dtype=np.float32),
        "df": np.array([0.5, 0.75, 1.25, 2.5], dtype=np.float32),
        "loc": np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32),
        "scale": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Broadcast between different dimensional shapes (2, 1) and (1, 3)
    input_dict = {
        "x": np.array([[1.0], [3.0]], dtype=np.float64),
        "df": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        "loc": np.array([[0.0]], dtype=np.float64),
        "scale": np.array([[1.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative loc parameter
    input_dict = {
        "x": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "df": np.array([3.0, 3.0, 3.0], dtype=np.float32),
        "loc": np.array([-5.0, -3.0, -1.0], dtype=np.float32),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.chi2.sf"] = chi2_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.chi2.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.chi2.sf'.")


check_valid('jax.scipy.stats.chi2.sf', generated_inputs['jax.scipy.stats.chi2.sf'], lib="jax", suffix=0)
