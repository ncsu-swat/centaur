
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_cauchy_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32
    list_of_inputs.append({
        "x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    })

    # Input 2: 0D arrays (scalars as numpy arrays)
    list_of_inputs.append({
        "x": np.array(1.5, dtype=np.float32),
        "loc": np.array(-0.5, dtype=np.float32),
        "scale": np.array(2.0, dtype=np.float32)
    })

    # Input 3: 2D arrays, float32, diverse values
    list_of_inputs.append({
        "x": np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float32),
        "loc": np.array([[0.5, -0.5], [1.5, -1.5]], dtype=np.float32),
        "scale": np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float32)
    })

    # Input 4: 1D arrays, float64
    list_of_inputs.append({
        "x": np.array([-10.0, 0.0, 10.0], dtype=np.float64),
        "loc": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "scale": np.array([0.5, 1.5, 2.5], dtype=np.float64)
    })

    # Input 5: Broadcasted shapes - x (3, 1), loc (1, 4), scale (3, 4)
    list_of_inputs.append({
        "x": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        "loc": np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.2, 1.5, 2.0], [1.0, 1.2, 1.5, 2.0], [1.0, 1.2, 1.5, 2.0]], dtype=np.float32)
    })

    # Input 6: Large dimensions (3D arrays)
    list_of_inputs.append({
        "x": np.random.randn(2, 3, 4).astype(np.float32),
        "loc": np.random.randn(2, 3, 4).astype(np.float32),
        "scale": np.abs(np.random.randn(2, 3, 4)).astype(np.float32) + 0.1
    })

    # Input 7: Small scale values (float32)
    list_of_inputs.append({
        "x": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([0.01, 0.05, 0.1], dtype=np.float32)
    })

    # Input 8: Very large values (float64)
    list_of_inputs.append({
        "x": np.array([100.0, 1000.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0], dtype=np.float64),
        "scale": np.array([10.0, 50.0], dtype=np.float64)
    })

    # Input 9: Broad-casting with scale scalar-like array
    list_of_inputs.append({
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "loc": np.array([0.0, 1.0], dtype=np.float32),
        "scale": np.array([2.5], dtype=np.float32)
    })

    # Input 10: Negative loc, diverse scale and float64
    list_of_inputs.append({
        "x": np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float64),
        "loc": np.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64),
        "scale": np.array([0.1, 0.5, 1.0, 2.0, 10.0], dtype=np.float64)
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.logpdf_1"] = jax_scipy_stats_cauchy_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logpdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logpdf_1'.")


check_valid('jax.scipy.stats.cauchy.logpdf', generated_inputs['jax.scipy.stats.cauchy.logpdf_1'], lib="jax", suffix=1)
