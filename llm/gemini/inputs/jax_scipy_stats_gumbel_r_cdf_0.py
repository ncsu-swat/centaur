
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_r_cdf_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D arrays
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: float64 1D arrays of same shape
    x = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    loc = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    scale = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 0D arrays (scalar arrays)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(-1.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: Broadcasting (x is 2D, loc is 1D, scale is 0D)
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Broadcasting (x is 1D, loc is 2D, scale is 2D)
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    loc = np.random.randn(2, 3).astype(np.float64)
    scale = (np.abs(np.random.randn(2, 3)) + 0.1).astype(np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: 3D arrays of same shape
    x = np.random.randn(2, 2, 2).astype(np.float32)
    loc = np.random.randn(2, 2, 2).astype(np.float32)
    scale = np.random.uniform(0.1, 5.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Large values (extreme boundaries)
    x = np.array([-100.0, -10.0, 10.0, 100.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Very small scale parameters
    x = np.random.randn(5).astype(np.float32)
    loc = np.zeros(5, dtype=np.float32)
    scale = np.array([1e-3, 1e-2, 1e-1, 1e-1, 1e-2], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Scale is a 1D array, others broadcasted
    x = np.array([1.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: Multi-dimensional inputs with float64
    x = np.random.uniform(-5, 5, (3, 1, 4)).astype(np.float64)
    loc = np.random.uniform(-1, 1, (1, 2, 4)).astype(np.float64)
    scale = np.random.uniform(0.5, 3.0, (3, 2, 1)).astype(np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return [copy.deepcopy(x) for x in list_of_inputs]

generated_inputs["jax.scipy.stats.gumbel_r.cdf"] = gumbel_r_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.cdf'.")


check_valid('jax.scipy.stats.gumbel_r.cdf', generated_inputs['jax.scipy.stats.gumbel_r.cdf'], lib="jax", suffix=0)
