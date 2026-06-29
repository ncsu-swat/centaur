
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_laplace_cdf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 2D float32 arrays with mixed positive/negative values
    x = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: Float64 high precision, 1D arrays
    x = np.array([1.5], dtype=np.float64)
    loc = np.array([0.5], dtype=np.float64)
    scale = np.array([2.5], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: Large 3D arrays with random values
    x = np.random.uniform(-5.0, 5.0, (2, 3, 4)).astype(np.float32)
    loc = np.random.uniform(-1.0, 1.0, (2, 3, 4)).astype(np.float32)
    scale = np.random.uniform(0.1, 5.0, (2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Broadcastable shapes (x: 2x1, loc: 1x3, scale: 2x3)
    x = np.array([[1.0], [-1.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.5, -0.5]], dtype=np.float32)
    scale = np.array([[1.0, 1.5, 2.0], [2.0, 1.0, 0.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Extremely small scale
    x = np.array([0.01, -0.01], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1e-5, 1e-5], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Large scale values
    x = np.array([100.0, -100.0], dtype=np.float32)
    loc = np.array([10.0, -10.0], dtype=np.float32)
    scale = np.array([1000.0, 1000.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Integer arrays (which will be implicitly converted)
    x = np.array([1, 2, -3], dtype=np.int32)
    loc = np.array([0, 1, -1], dtype=np.int32)
    scale = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: 4D array with standard normal-like values
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: Negative offset loc, positive x, and positive scale
    x = np.array([5.0], dtype=np.float32)
    loc = np.array([-5.0], dtype=np.float32)
    scale = np.array([0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.laplace.cdf_1"] = generate_laplace_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.laplace.cdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.laplace.cdf_1'.")


check_valid('jax.scipy.stats.laplace.cdf', generated_inputs['jax.scipy.stats.laplace.cdf_1'], lib="jax", suffix=1)
