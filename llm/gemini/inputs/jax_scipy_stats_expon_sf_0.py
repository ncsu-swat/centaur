
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_expon_sf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, scale > 0
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 1D arrays with float64, non-zero loc
    x = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    loc = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 2D arrays, float32
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    loc = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    scale = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 3D arrays, float64
    x = np.random.uniform(0.5, 5.0, size=(2, 2, 2)).astype(np.float64)
    loc = np.random.uniform(0.0, 0.4, size=(2, 2, 2)).astype(np.float64)
    scale = np.random.uniform(0.5, 2.0, size=(2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: 0D arrays (scalar representation)
    x = np.array(2.5, dtype=np.float32)
    loc = np.array(0.5, dtype=np.float32)
    scale = np.array(1.2, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Broadcasting - x is 2D, loc and scale are 1D
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    loc = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    scale = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Broadcasting - x is 2D, loc and scale are 0D (scalars)
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: High dimensions (4D)
    x = np.random.uniform(1.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2, 2), dtype=np.float32) * 3.0
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Negative loc, positive x
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    scale = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: x is below loc (survival function should evaluate to 1.0)
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 11: Very large scale parameter
    x = np.array([100.0, 200.0], dtype=np.float64)
    loc = np.array([10.0, 20.0], dtype=np.float64)
    scale = np.array([1000.0, 2000.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.expon.sf"] = jax_scipy_stats_expon_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.sf'.")


check_valid('jax.scipy.stats.expon.sf', generated_inputs['jax.scipy.stats.expon.sf'], lib="jax", suffix=0)
