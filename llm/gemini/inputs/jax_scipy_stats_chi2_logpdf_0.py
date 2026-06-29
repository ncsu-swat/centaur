
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_chi2_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0-D tensors (scalars as arrays), standard float32
    x = np.array(2.5, dtype=np.float32)
    df = np.array(3.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 2: 1-D arrays, float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    df = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 3: 1-D arrays, float64, non-zero loc and scale
    x = np.array([5.0, 6.0, 7.0], dtype=np.float64)
    df = np.array([4.0, 4.0, 4.0], dtype=np.float64)
    loc = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 4: 2-D arrays, float32
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    df = np.array([[2.5, 2.5], [2.5, 2.5]], dtype=np.float32)
    loc = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    scale = np.array([[1.5, 1.5], [1.5, 1.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 5: 3-D arrays, random values
    x = np.abs(np.random.randn(2, 3, 4).astype(np.float32)) + 1.0
    df = np.random.uniform(1.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.random.uniform(0.5, 2.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 6: Broadcasting shapes
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    df = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 7: Negative values for x
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    df = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 8: High dimensions with float64
    x = np.abs(np.random.randn(1, 5, 1).astype(np.float64)) + 2.0
    df = np.array([[[15.0]]], dtype=np.float64)
    loc = np.array([[[1.0]]], dtype=np.float64)
    scale = np.array([[[0.5]]], dtype=np.float64)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 9: Large df parameter
    x = np.array([100.0, 200.0], dtype=np.float32)
    df = np.array([100.0, 150.0], dtype=np.float32)
    loc = np.array([10.0, 10.0], dtype=np.float32)
    scale = np.array([5.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 10: Fractional df and offset loc
    x = np.array([0.5, 1.5], dtype=np.float32)
    df = np.array([0.5, 1.2], dtype=np.float32)
    loc = np.array([-1.0, -1.0], dtype=np.float32)
    scale = np.array([0.1, 0.2], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.chi2.logpdf"] = jax_scipy_stats_chi2_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.chi2.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.chi2.logpdf'.")


check_valid('jax.scipy.stats.chi2.logpdf', generated_inputs['jax.scipy.stats.chi2.logpdf'], lib="jax", suffix=0)
