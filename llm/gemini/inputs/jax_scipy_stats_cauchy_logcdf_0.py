
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_cauchy_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    x = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: float64 arrays with non-zero loc and scale
    x = np.array([1.5, -2.5, 3.0], dtype=np.float64)
    loc = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    scale = np.array([2.0, 0.5, 1.5], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 2D arrays
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.zeros((3, 4), dtype=np.float32)
    scale = np.ones((3, 4), dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: Broadcasting: x is 2D, loc is 1D, scale is scalar (0D)
    x = np.random.randn(2, 3).astype(np.float32)
    loc = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Large positive values of x
    x = np.array([100.0, 1000.0, 10000.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Large negative values of x
    x = np.array([-100.0, -1000.0, -10000.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Scalar-like tensors (0D arrays)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(-0.5, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: 3D arrays with float64
    x = np.random.randn(2, 2, 2).astype(np.float64)
    loc = np.random.randn(2, 2, 2).astype(np.float64)
    scale = np.abs(np.random.randn(2, 2, 2)).astype(np.float64) + 0.1
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Scale contains very small positive values
    x = np.array([0.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1e-5, 1e-4], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: 4D arrays, float32
    x = np.random.randn(2, 3, 2, 1).astype(np.float32)
    loc = np.zeros((2, 3, 2, 1), dtype=np.float32)
    scale = np.ones((2, 3, 2, 1), dtype=np.float32) * 5.0
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.logcdf"] = jax_scipy_stats_cauchy_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logcdf'.")


check_valid('jax.scipy.stats.cauchy.logcdf', generated_inputs['jax.scipy.stats.cauchy.logcdf'], lib="jax", suffix=0)
