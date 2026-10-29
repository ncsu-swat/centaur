
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def t_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "df": np.array(5.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with float64
    input_dict = {
        "x": np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float64),
        "df": np.array([2.0, 3.0, 5.0, 10.0, 30.0], dtype=np.float64),
        "loc": np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float64),
        "scale": np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting 1D x with scalar df, loc, scale
    input_dict = {
        "x": np.linspace(-5.0, 5.0, 20, dtype=np.float32),
        "df": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays of same shape
    input_dict = {
        "x": np.random.randn(3, 4).astype(np.float32),
        "df": np.random.uniform(1.0, 10.0, size=(3, 4)).astype(np.float32),
        "loc": np.random.randn(3, 4).astype(np.float32),
        "scale": np.random.uniform(0.1, 5.0, size=(3, 4)).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting 2D x with 1D loc and scale
    input_dict = {
        "x": np.random.randn(4, 5).astype(np.float64),
        "df": np.array([[2.0], [5.0], [10.0], [20.0]], dtype=np.float64),
        "loc": np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float64),
        "scale": np.array([1.0, 2.0, 0.5, 1.5, 3.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays
    input_dict = {
        "x": np.random.randn(2, 3, 4).astype(np.float32),
        "df": np.full((2, 3, 4), 4.0, dtype=np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
        "scale": np.ones((2, 3, 4), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small degrees of freedom (df < 1)
    input_dict = {
        "x": np.array([-10.0, 0.0, 10.0], dtype=np.float32),
        "df": np.array([0.5, 0.75, 0.25], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large degrees of freedom approximating normal distribution
    input_dict = {
        "x": np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=np.float64),
        "df": np.array([1e4, 1e5, 1e6, 1e7, 1e8], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-standard loc and large scale
    input_dict = {
        "x": np.array([100.0, 150.0, 200.0], dtype=np.float32),
        "df": np.array([5.0, 5.0, 5.0], dtype=np.float32),
        "loc": np.array([100.0, 100.0, 100.0], dtype=np.float32),
        "scale": np.array([50.0, 50.0, 50.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D tensor inputs
    input_dict = {
        "x": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "df": np.random.uniform(1.0, 30.0, size=(2, 2, 2, 2)).astype(np.float32),
        "loc": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "scale": np.random.uniform(0.5, 2.0, size=(2, 2, 2, 2)).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Complex broadcasting across different axes
    input_dict = {
        "x": np.linspace(-2.0, 2.0, 6, dtype=np.float32).reshape(6, 1, 1),
        "df": np.array([1.0, 2.0, 5.0, 10.0], dtype=np.float32).reshape(1, 4, 1),
        "loc": np.array([0.0, 1.0, -1.0], dtype=np.float32).reshape(1, 1, 3),
        "scale": np.array([0.5, 1.0, 2.0], dtype=np.float32).reshape(1, 1, 3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.t.pdf"] = t_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.t.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.t.pdf'.")


check_valid('jax.scipy.stats.t.pdf', generated_inputs['jax.scipy.stats.t.pdf'], lib="jax", suffix=0)
