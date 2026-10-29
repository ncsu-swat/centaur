
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logistic_isf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, standard logistic
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 1D arrays, float32
    x = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    loc = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    scale = np.array([1.0, 0.5, 2.0, 1.5, 0.8], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 1D with float64
    x = np.array([0.01, 0.05, 0.95, 0.99], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 2D arrays, matching shapes
    x = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    loc = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: 2D broadcast with 1D
    x = np.linspace(0.1, 0.9, 6, dtype=np.float32).reshape(2, 3)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([[1.0], [2.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: 3D arrays
    x = np.full((2, 3, 4), 0.3, dtype=np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Broadcasting 0D loc and scale to 2D x
    x = np.array([[0.05, 0.15, 0.25], [0.75, 0.85, 0.95]], dtype=np.float64)
    loc = np.array(-5.0, dtype=np.float64)
    scale = np.array(0.25, dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: 1D broadcast against 0D x
    x = np.array(0.4, dtype=np.float32)
    loc = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    scale = np.array([0.1, 1.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Large negative loc and large scale values
    x = np.array([0.3, 0.7], dtype=np.float32)
    loc = np.array([-100.0, -200.0], dtype=np.float32)
    scale = np.array([50.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: 4D tensor with singleton dimensions for broadcasting
    x = np.array([[[[0.2]], [[0.8]]]], dtype=np.float32)
    loc = np.array([[[[0.5, -0.5]]]], dtype=np.float32)
    scale = np.array([[[[1.0, 2.0]]]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs


generated_inputs["jax.scipy.stats.logistic.isf"] = logistic_isf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.isf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.isf'.")


check_valid('jax.scipy.stats.logistic.isf', generated_inputs['jax.scipy.stats.logistic.isf'], lib="jax", suffix=0)
