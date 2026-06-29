
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_gamma_sf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, standard gamma
    x = np.array([0.5, 1.0, 2.5], dtype=np.float32)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 2: 2D arrays, float32, positive parameters
    x = np.random.uniform(1.0, 5.0, size=(2, 3)).astype(np.float32)
    a = np.random.uniform(0.5, 3.0, size=(2, 3)).astype(np.float32)
    loc = np.zeros((2, 3), dtype=np.float32)
    scale = np.random.uniform(0.5, 2.0, size=(2, 3)).astype(np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 3: 0D arrays (scalars), float64
    x = np.array(1.5, dtype=np.float64)
    a = np.array(2.5, dtype=np.float64)
    loc = np.array(-0.5, dtype=np.float64)
    scale = np.array(1.2, dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 4: 3D arrays, float32
    x = np.random.uniform(2.0, 10.0, size=(2, 2, 2)).astype(np.float32)
    a = np.random.uniform(1.0, 5.0, size=(2, 2, 2)).astype(np.float32)
    loc = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    scale = np.random.uniform(0.1, 2.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 5: float64 1D arrays
    x = np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float64)
    a = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 6: Broadcasting shapes
    x = np.array([[1.0], [2.0]], dtype=np.float32)
    a = np.array([1.5, 2.5], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 7: Large scale values
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    a = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 8: Small scale and small shape parameters
    x = np.array([0.01, 0.02], dtype=np.float32)
    a = np.array([0.1, 0.1], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([0.1, 0.1], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 9: Negative loc parameter
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    a = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 10: 4D arrays, float32
    x = np.random.uniform(1.0, 3.0, size=(2, 2, 1, 3)).astype(np.float32)
    a = np.random.uniform(1.0, 2.0, size=(2, 2, 1, 3)).astype(np.float32)
    loc = np.zeros((2, 2, 1, 3), dtype=np.float32)
    scale = np.ones((2, 2, 1, 3), dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.gamma.sf"] = jax_scipy_stats_gamma_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gamma.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gamma.sf'.")


check_valid('jax.scipy.stats.gamma.sf', generated_inputs['jax.scipy.stats.gamma.sf'], lib="jax", suffix=0)
