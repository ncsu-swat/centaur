
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logistic_ppf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard logistic
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 1D array, uniform quantiles, default loc and scale
    x = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 1D arrays with matching dimensions, negative loc
    x = np.array([0.05, 0.2, 0.8, 0.95], dtype=np.float64)
    loc = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float64)
    scale = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 2D array, float32, broadcasting scalar loc and scale
    x = np.random.uniform(0.1, 0.9, size=(3, 4)).astype(np.float32)
    loc = np.array(5.0, dtype=np.float32)
    scale = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: 2D arrays with row/column broadcasting
    x = np.random.uniform(0.01, 0.99, size=(4, 1)).astype(np.float64)
    loc = np.random.randn(1, 5).astype(np.float64)
    scale = np.random.uniform(0.5, 3.0, size=(4, 5)).astype(np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: 3D array, float32, with positive loc and small scale
    x = np.random.uniform(0.2, 0.8, size=(2, 3, 4)).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.full((2, 3, 4), 0.1, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: 3D broadcasting across dimensions
    x = np.random.uniform(0.1, 0.9, size=(2, 1, 4)).astype(np.float32)
    loc = np.random.randn(1, 3, 1).astype(np.float32)
    scale = np.random.uniform(0.5, 2.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Extreme tail quantiles, float64
    x = np.array([1e-5, 1e-3, 0.5, 1.0 - 1e-3, 1.0 - 1e-5], dtype=np.float64)
    loc = np.array([0.0], dtype=np.float64)
    scale = np.array([1.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: 4D arrays
    x = np.random.uniform(0.05, 0.95, size=(2, 2, 3, 3)).astype(np.float32)
    loc = np.random.uniform(-5.0, 5.0, size=(2, 2, 3, 3)).astype(np.float32)
    scale = np.random.uniform(0.1, 5.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: 1D array with single element tensors
    x = np.array([0.7], dtype=np.float32)
    loc = np.array([-10.0], dtype=np.float32)
    scale = np.array([0.01], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.scipy.stats.logistic.ppf"] = logistic_ppf_inputs()

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
