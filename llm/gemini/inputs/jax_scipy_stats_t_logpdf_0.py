
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def t_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard Student's t
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "df": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with negative x values
    input_dict = {
        "x": np.array([-2.5, -1.0, 0.0, 1.5, 3.0], dtype=np.float32),
        "df": np.array([1.0, 2.0, 5.0, 10.0, 30.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float64 dtype
    input_dict = {
        "x": np.array([[-1.0, 0.5], [1.2, -0.8]], dtype=np.float64),
        "df": np.array([[2.5, 3.5], [4.5, 5.5]], dtype=np.float64),
        "loc": np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float64),
        "scale": np.array([[0.5, 1.5], [2.0, 0.8]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting shapes (3, 1) and (1, 4)
    input_dict = {
        "x": np.array([[-1.0], [0.0], [1.0]], dtype=np.float32),
        "df": np.array([[1.0, 2.0, 5.0, 10.0]], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays
    input_dict = {
        "x": np.ones((2, 3, 4), dtype=np.float32) * 1.5,
        "df": np.ones((2, 3, 4), dtype=np.float32) * 8.0,
        "loc": np.ones((2, 3, 4), dtype=np.float32) * -0.5,
        "scale": np.ones((2, 3, 4), dtype=np.float32) * 2.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Non-zero negative locations and non-unit scales
    input_dict = {
        "x": np.array([-10.0, -5.0, -2.0], dtype=np.float32),
        "df": np.array([4.0, 4.0, 4.0], dtype=np.float32),
        "loc": np.array([-5.0, -5.0, -5.0], dtype=np.float32),
        "scale": np.array([0.25, 0.5, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Fractional degrees of freedom (df < 1)
    input_dict = {
        "x": np.array([0.0, 0.1, 0.5, 1.0], dtype=np.float64),
        "df": np.array([0.2, 0.5, 0.7, 0.9], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large degrees of freedom approaching Gaussian
    input_dict = {
        "x": np.array([0.0, 1.96, -1.96], dtype=np.float32),
        "df": np.array([1000.0, 5000.0, 10000.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensors
    input_dict = {
        "x": np.zeros((1, 2, 2, 1), dtype=np.float32),
        "df": np.full((1, 2, 2, 1), 6.0, dtype=np.float32),
        "loc": np.full((1, 2, 2, 1), 0.5, dtype=np.float32),
        "scale": np.full((1, 2, 2, 1), 1.5, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Extreme values in x
    input_dict = {
        "x": np.array([-100.0, -50.0, 50.0, 100.0], dtype=np.float64),
        "df": np.array([3.0, 5.0, 5.0, 3.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.t.logpdf"] = t_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.t.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.t.logpdf'.")


check_valid('jax.scipy.stats.t.logpdf', generated_inputs['jax.scipy.stats.t.logpdf'], lib="jax", suffix=0)
