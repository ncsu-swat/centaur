
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gamma_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 2: 2D float32 arrays
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    loc = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    scale = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 3: float64 with negative loc values
    x = np.array([10.0, 20.0], dtype=np.float64)
    a = np.array([5.0, 5.0], dtype=np.float64)
    loc = np.array([-1.0, -2.0], dtype=np.float64)
    scale = np.array([2.0, 3.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 4: Scalar-like tensors (0D)
    x = np.array(1.5, dtype=np.float32)
    a = np.array(0.5, dtype=np.float32)
    loc = np.array(-0.5, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 5: Broadcastable shapes (2x1 and 1x3)
    x = np.array([[1.0], [2.0]], dtype=np.float32)
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    loc = np.array([[0.0]], dtype=np.float32)
    scale = np.array([[1.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 6: 3D float32 arrays
    x = np.ones((2, 2, 2), dtype=np.float32) * 5.0
    a = np.ones((2, 2, 2), dtype=np.float32) * 2.0
    loc = np.ones((2, 2, 2), dtype=np.float32) * 1.0
    scale = np.ones((2, 2, 2), dtype=np.float32) * 0.5
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 7: Very small shape parameter 'a'
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    a = np.array([0.01, 0.05, 0.1], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 8: Large scale and shape parameters
    x = np.array([100.0, 200.0], dtype=np.float32)
    a = np.array([50.0, 100.0], dtype=np.float32)
    loc = np.array([10.0, 20.0], dtype=np.float32)
    scale = np.array([10.0, 20.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 9: Shifted location and scaling values
    x = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([-5.0, -10.0, -15.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 10: 1D float64 with higher precision
    x = np.array([1.5e2, 2.5e2], dtype=np.float64)
    a = np.array([1e2, 1.5e2], dtype=np.float64)
    loc = np.array([0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 11: 4D arrays for dimensionality variety
    x = np.ones((2, 2, 2, 2), dtype=np.float32) * 4.0
    a = np.ones((2, 2, 2, 2), dtype=np.float32) * 3.0
    loc = np.ones((2, 2, 2, 2), dtype=np.float32) * 2.0
    scale = np.ones((2, 2, 2, 2), dtype=np.float32) * 1.5
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.gamma.logpdf"] = gamma_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gamma.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gamma.logpdf'.")


check_valid('jax.scipy.stats.gamma.logpdf', generated_inputs['jax.scipy.stats.gamma.logpdf'], lib="jax", suffix=0)
