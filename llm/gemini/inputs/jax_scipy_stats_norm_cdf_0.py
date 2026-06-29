
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_norm_cdf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays, positive scale
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 arrays, non-zero mean and std
    x = np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float32)
    loc = np.array([[0.5, -0.5], [0.0, 1.0]], dtype=np.float32)
    scale = np.array([[1.5, 0.5], [2.0, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 arrays, 1D
    x = np.array([0.5, 1.5, -0.5], dtype=np.float64)
    loc = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    scale = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar-like 0D arrays
    x = np.array(0.0, dtype=np.float32)
    loc = np.array(-1.0, dtype=np.float32)
    scale = np.array(2.5, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large dimension 3D arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting shapes
    x = np.random.randn(3, 3).astype(np.float32)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small positive scale values
    x = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([0.01, 0.05, 0.1], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large scale values
    x = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
    loc = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    scale = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Extreme values of x
    x = np.array([-1e5, 1e5], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D arrays with randomly generated scales
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    loc = np.random.randn(2, 2, 2, 2).astype(np.float32)
    scale = np.abs(np.random.randn(2, 2, 2, 2).astype(np.float32)) + 0.1
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.norm.cdf"] = jax_scipy_stats_norm_cdf_inputs()

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
