
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_uniform_cdf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 tensors, standard uniform [0, 1]
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 2D float32 tensors, uniform [1, 3] (loc=1, scale=2)
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    loc = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    scale = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: float64 tensors with negative location, large scale
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    loc = np.array([-5.0], dtype=np.float64)
    scale = np.array([10.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 0D tensors (scalars represented as tensors)
    x = np.array(0.3, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Broadcasting - x is 2D, loc and scale are 1D
    x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    loc = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    scale = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Values completely outside the support (below and above)
    x = np.array([-2.0, -1.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Small scale values with float64 precision
    x = np.array([1.001, 1.002, 1.003], dtype=np.float64)
    loc = np.array([1.000], dtype=np.float64)
    scale = np.array([0.005], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: 3D tensors
    x = np.ones((2, 2, 2), dtype=np.float32) * 0.5
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32) * 2.0
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Large negative values for location and scale
    x = np.array([-105.0, -100.0, -95.0], dtype=np.float32)
    loc = np.array([-100.0], dtype=np.float32)
    scale = np.array([10.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: Broadcasting - loc and scale are 2D, x is 1D
    x = np.array([0.5, 1.5], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.uniform.cdf"] = jax_scipy_stats_uniform_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.uniform.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.uniform.cdf'.")


check_valid('jax.scipy.stats.uniform.cdf', generated_inputs['jax.scipy.stats.uniform.cdf'], lib="jax", suffix=0)
