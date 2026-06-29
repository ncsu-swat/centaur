
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_laplace_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D arrays, float32
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 2D arrays, float64
    x = np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float64)
    loc = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    scale = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 0D arrays (scalars as tensors)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(-0.5, dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 3D arrays, float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.abs(np.random.randn(2, 3, 4)).astype(np.float32) + 0.1
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Broadcastable shapes (2D x, 1D loc, 0D scale)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    loc = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Large values for loc and x, float64
    x = np.array([100.0, -100.0], dtype=np.float64)
    loc = np.array([50.0, -50.0], dtype=np.float64)
    scale = np.array([10.0, 10.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Very small positive scale values
    x = np.array([0.1, -0.1], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([0.01, 0.02], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: High dimensional 4D arrays, float32
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    loc = np.ones((2, 2, 2, 2), dtype=np.float32) * -1.0
    scale = np.ones((2, 2, 2, 2), dtype=np.float32) * 0.5
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Mixed signs and float64
    x = np.array([-1.5, 2.5, -3.5], dtype=np.float64)
    loc = np.array([1.5, -2.5, 3.5], dtype=np.float64)
    scale = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: Broad cast with 1D x and 2D loc/scale
    x = np.array([1.0, 2.0], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.laplace.logpdf"] = jax_scipy_stats_laplace_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.laplace.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.laplace.logpdf'.")


check_valid('jax.scipy.stats.laplace.logpdf', generated_inputs['jax.scipy.stats.laplace.logpdf'], lib="jax", suffix=0)
