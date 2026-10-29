
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (float32)
    x = np.array(0.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 2: 1D arrays with negative values and float64
    x = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float64)
    loc = np.array([0.0, -0.5, 0.0, 0.5, 0.0], dtype=np.float64)
    scale = np.array([1.0, 2.0, 0.5, 1.5, 3.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 3: 2D arrays (matching shapes)
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.ones((3, 4), dtype=np.float32) * 0.5
    scale = np.abs(np.random.randn(3, 4)).astype(np.float32) + 0.1
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 4: Broadcasting 1D to 2D
    x = np.linspace(-3.0, 3.0, 10, dtype=np.float32).reshape(10, 1)
    loc = np.array([0.0, 1.0, -1.0], dtype=np.float32).reshape(1, 3)
    scale = np.array([1.0, 0.5, 2.0], dtype=np.float32).reshape(1, 3)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 5: 3D arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 6: Integer tensor inputs (converted internally)
    x = np.array([-3, -1, 0, 2, 5], dtype=np.int32)
    loc = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    scale = np.array([1, 1, 2, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 7: Float32 1D array
    x = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 8: 4D arrays
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    loc = np.random.randn(2, 2, 3, 3).astype(np.float32)
    scale = np.abs(np.random.randn(2, 2, 3, 3)).astype(np.float32) + 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 9: Large positive and negative values (extremes)
    x = np.array([-100.0, -10.0, 0.0, 10.0, 100.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 10: Broadcasting 0D loc and scale across 2D x
    x = np.random.randn(5, 5).astype(np.float32)
    loc = np.array(2.0, dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.norm.cdf"] = norm_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.cdf'.")


check_valid('jax.scipy.stats.norm.cdf', generated_inputs['jax.scipy.stats.norm.cdf'], lib="jax", suffix=0)
