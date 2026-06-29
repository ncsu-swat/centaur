
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_norm_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 tensors
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 1D float64 tensors with positive and negative shifts
    x = np.array([-2.5, 3.5], dtype=np.float64)
    loc = np.array([1.0, -1.0], dtype=np.float64)
    scale = np.array([0.5, 2.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 2D tensors, float32
    x = np.array([[-1.0, 2.0], [0.5, -0.5]], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    scale = np.array([[1.0, 0.5], [1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: Large positive values to test asymptotic behavior
    x = np.array([10.0, 50.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Large negative values
    x = np.array([-10.0, -50.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Broadcasting scenario (0D/scalar tensors for loc and scale)
    x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: 1D float32 tensors with different scale
    x = np.array([0.1, -0.1], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([2.5, 2.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: 3D tensors, float32
    x = np.random.randn(2, 2, 2).astype(np.float32)
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Small scale values (narrow distributions)
    x = np.array([0.01, -0.01], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([0.001, 0.001], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: 4D tensors, float32
    x = np.random.randn(2, 1, 3, 3).astype(np.float32)
    loc = np.random.randn(2, 1, 3, 3).astype(np.float32)
    scale = np.abs(np.random.randn(2, 1, 3, 3).astype(np.float32)) + 0.1
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.norm.logcdf"] = jax_scipy_stats_norm_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.logcdf'.")


check_valid('jax.scipy.stats.norm.logcdf', generated_inputs['jax.scipy.stats.norm.logcdf'], lib="jax", suffix=0)
