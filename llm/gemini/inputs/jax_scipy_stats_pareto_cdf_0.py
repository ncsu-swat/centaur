
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_pareto_cdf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays, standard loc=0, scale=1
    x = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 2: 0D scalar arrays, float64
    x = np.array(3.5, dtype=np.float64)
    b = np.array(1.5, dtype=np.float64)
    loc = np.array(0.5, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 3: 2D arrays, float32, positive b
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    b = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 4: Broadcast shapes, x is 2D, b is 1D, loc/scale are 1D
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    b = np.array([1.0, 2.0], dtype=np.float32)
    loc = np.array([0.5, 1.0], dtype=np.float32)
    scale = np.array([0.5, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 5: High dimensions (3D arrays)
    x = np.ones((2, 2, 2), dtype=np.float32) * 10.0
    b = np.ones((2, 2, 2), dtype=np.float32) * 0.5
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32) * 2.0
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 6: x below loc + scale (CDF should be 0)
    x = np.array([-1.0, 0.0, 0.5], dtype=np.float32)
    b = np.array([2.5, 2.5, 2.5], dtype=np.float32)
    loc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 7: Negative loc and positive scale, float32
    x = np.array([-1.0, 1.0, 3.0], dtype=np.float32)
    b = np.array([1.2, 1.2, 1.2], dtype=np.float32)
    loc = np.array([-3.0, -3.0, -3.0], dtype=np.float32)
    scale = np.array([1.5, 1.5, 1.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 8: b parameter is very small (but positive)
    x = np.array([100.0, 1000.0], dtype=np.float32)
    b = np.array([1e-3, 1e-4], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 9: Large scale parameter, float64
    x = np.array([200.0, 300.0], dtype=np.float64)
    b = np.array([2.0, 3.0], dtype=np.float64)
    loc = np.array([10.0, 10.0], dtype=np.float64)
    scale = np.array([100.0, 100.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 10: Mixed dimensions, standard broadcasting
    x = np.array([[[2.0]]], dtype=np.float32)
    b = np.array([1.0, 2.0], dtype=np.float32)
    loc = np.array([[0.0], [0.5]], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.pareto.cdf"] = jax_scipy_stats_pareto_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.cdf'.")


check_valid('jax.scipy.stats.pareto.cdf', generated_inputs['jax.scipy.stats.pareto.cdf'], lib="jax", suffix=0)
