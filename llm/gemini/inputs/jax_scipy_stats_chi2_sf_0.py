
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_chi2_sf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    x = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    df = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 2: 2D float32 arrays
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    df = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.5], [-0.5, 0.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.5], [0.5, 2.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 3: Scalar-like 0D float64 arrays
    x = np.array(3.0, dtype=np.float64)
    df = np.array(5.0, dtype=np.float64)
    loc = np.array(1.0, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 4: 3D float32 arrays
    x = np.ones((2, 2, 2), dtype=np.float32) * 4.0
    df = np.ones((2, 2, 2), dtype=np.float32) * 3.0
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 5: Negative x values (sf should be 1.0 if x < loc)
    x = np.array([-1.0, -2.0, -5.0], dtype=np.float32)
    df = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 6: Large degrees of freedom, float64
    x = np.array([50.0, 100.0], dtype=np.float64)
    df = np.array([45.0, 90.0], dtype=np.float64)
    loc = np.array([0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 7: Broadcasting sizes (x: 2D, df: 1D, loc: 0D, scale: 0D)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    df = np.array([2.0, 3.0], dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 8: Very small scale values
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    df = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([0.01, 0.02, 0.05], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 9: High dimensions (4D)
    x = np.random.uniform(5.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    df = np.random.uniform(1.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    scale = np.random.uniform(0.5, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 10: Float64 high precision
    x = np.array([10.5, 20.5], dtype=np.float64)
    df = np.array([8.0, 15.0], dtype=np.float64)
    loc = np.array([1.0, 2.0], dtype=np.float64)
    scale = np.array([1.5, 0.8], dtype=np.float64)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.chi2.sf"] = jax_scipy_stats_chi2_sf_inputs()

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
