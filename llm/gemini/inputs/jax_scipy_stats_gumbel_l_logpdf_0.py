
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_l_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard values
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 2: 1D arrays of matching shapes
    x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0], dtype=np.float32)
    loc = np.array([0.0, -1.0, 0.5, 2.0, -0.5], dtype=np.float32)
    scale = np.array([1.0, 0.5, 2.0, 1.5, 0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 3: Float64 2D arrays
    x = np.linspace(-3.0, 3.0, 12, dtype=np.float64).reshape(3, 4)
    loc = np.zeros((3, 4), dtype=np.float64)
    scale = np.ones((3, 4), dtype=np.float64) * 1.5
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 4: Broadcasting 1D loc and scale with 2D x
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    loc = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    scale = np.array([[0.5], [1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 5: 3D arrays
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    loc = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    scale = np.random.uniform(0.5, 3.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 6: Highly negative x values
    x = np.array([-100.0, -50.0, -20.0, -10.0], dtype=np.float32)
    loc = np.array([-10.0, -5.0, 0.0, 5.0], dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 7: Small positive scale values
    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    loc = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    scale = np.array([1e-3, 1e-2, 1e-1], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 8: 4D tensors with broadcasting scalar-like tensor
    x = np.ones((2, 2, 3, 3), dtype=np.float32)
    loc = np.array([[[[0.0]]]], dtype=np.float32)
    scale = np.array([[[[2.5]]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 9: Negative loc and positive x
    x = np.array([1.0, 2.5, 5.0, 10.0], dtype=np.float32)
    loc = np.array([-10.0, -8.0, -6.0, -4.0], dtype=np.float32)
    scale = np.array([3.0, 2.0, 1.0, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 10: Empty arrays
    x = np.empty((0, 2), dtype=np.float32)
    loc = np.empty((0, 2), dtype=np.float32)
    scale = np.empty((0, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 11: Broadcasting 3D with 1D
    x = np.zeros((4, 3, 2), dtype=np.float64)
    loc = np.array([0.0, 1.0], dtype=np.float64)
    scale = np.array([1.0, 2.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gumbel_l.logpdf"] = gumbel_l_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.logpdf'.")


check_valid('jax.scipy.stats.gumbel_l.logpdf', generated_inputs['jax.scipy.stats.gumbel_l.logpdf'], lib="jax", suffix=0)
