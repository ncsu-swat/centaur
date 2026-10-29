
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays with float32
    x = np.array([0.0, 1.0, -1.0, 2.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 2: 0D scalar arrays
    x = np.array(1.5, dtype=np.float64)
    loc = np.array(-0.5, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 3: 2D arrays with matching shapes
    x = np.array([[-2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 4: Broadcasting 1D loc and scale across 2D x
    x = np.zeros((3, 4), dtype=np.float64)
    loc = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    scale = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 5: 3D arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32) * 1.5
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 6: Large positive and negative values in x
    x = np.array([-100.0, -10.0, 0.0, 10.0, 100.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 7: Broadcasting scalar loc and 1D scale with 2D x
    x = np.ones((2, 3), dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 1.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 8: Integer arrays (converted to float inside JAX, but input as int tensor)
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    loc = np.array([0, 1, 2, 3], dtype=np.int32)
    scale = np.array([1, 2, 1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 9: 4D arrays with small scale
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.full((2, 2, 2, 2), 0.1, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 10: Asymmetric negative loc and varying scale
    x = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float64)
    loc = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    scale = np.array([10.0, 5.0, 1.0, 5.0, 10.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 11: 1D array with single element
    x = np.array([3.14159], dtype=np.float64)
    loc = np.array([0.0], dtype=np.float64)
    scale = np.array([0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.logpdf_1"] = cauchy_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logpdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logpdf_1'.")


check_valid('jax.scipy.stats.cauchy.logpdf', generated_inputs['jax.scipy.stats.cauchy.logpdf_1'], lib="jax", suffix=1)
