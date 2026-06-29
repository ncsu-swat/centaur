
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_logistic_ppf_inputs():
    list_of_inputs = []

    # Input 1: 1D standard case, float32
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 2D arrays, negative loc, scale > 1
    x = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    loc = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    scale = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: float64, 3D arrays
    x = np.random.uniform(0.1, 0.9, size=(2, 2, 2)).astype(np.float64)
    loc = np.random.normal(0, 1, size=(2, 2, 2)).astype(np.float64)
    scale = np.random.uniform(0.5, 2.0, size=(2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: Scalar arrays (0D)
    x = np.array(0.25, dtype=np.float32)
    loc = np.array(5.0, dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Broadcastable shapes (x: (3, 1), loc: (1, 3), scale: (1, 1))
    x = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)
    loc = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    scale = np.array([[1.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Large scale values, float64
    x = np.linspace(0.01, 0.99, 5).astype(np.float64)
    loc = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    scale = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Very small scale (precision test)
    x = np.array([0.05, 0.95], dtype=np.float32)
    loc = np.array([-0.5, 0.5], dtype=np.float32)
    scale = np.array([1e-3, 1e-3], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Extreme probabilities (near 0 and 1)
    x = np.array([1e-6, 0.5, 1 - 1e-6], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: float16 type inputs
    x = np.array([0.3, 0.7], dtype=np.float16)
    loc = np.array([1.0, -1.0], dtype=np.float16)
    scale = np.array([0.5, 2.0], dtype=np.float16)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: High-dimensional array
    x = np.random.uniform(0.01, 0.99, size=(1, 2, 3, 4)).astype(np.float32)
    loc = np.zeros((1, 2, 3, 4), dtype=np.float32)
    scale = np.ones((1, 2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.ppf"] = jax_scipy_stats_logistic_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.ppf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.ppf'.")


check_valid('jax.scipy.stats.logistic.ppf', generated_inputs['jax.scipy.stats.logistic.ppf'], lib="jax", suffix=0)
