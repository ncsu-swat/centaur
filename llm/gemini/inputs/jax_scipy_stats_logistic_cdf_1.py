
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_logistic_cdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays (float32)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D arrays (scalars as tensors)
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(0.5, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float64, negative loc
    x = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    loc = np.array([[-0.5, -0.5], [-0.5, -0.5]], dtype=np.float64)
    scale = np.array([[1.5, 1.5], [1.5, 1.5]], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting, x is 2D, loc/scale are 1D
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting, x is 1D, loc/scale are 3D
    x = np.array([0.0, 1.0], dtype=np.float32)
    loc = np.random.randn(2, 2, 2).astype(np.float32)
    scale = np.random.uniform(0.1, 5.0, size=(2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large scales and extreme values
    x = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small scales and precise float64
    x = np.array([-0.01, 0.0, 0.01], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D arrays, complex shape
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    loc = np.random.randn(2, 2, 2, 2).astype(np.float32)
    scale = np.random.uniform(0.5, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element 1D arrays
    x = np.array([5.0], dtype=np.float32)
    loc = np.array([-5.0], dtype=np.float32)
    scale = np.array([0.1], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed precision types for loc and x
    x = np.array([1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.cdf_1"] = jax_scipy_stats_logistic_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.cdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.cdf_1'.")


check_valid('jax.scipy.stats.logistic.cdf', generated_inputs['jax.scipy.stats.logistic.cdf_1'], lib="jax", suffix=1)
