
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def beta_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D arrays, standard support [0, 1], float32
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 2: 2D arrays, float32, standard support
    x = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    a = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    b = np.array([[3.5, 2.5], [1.5, 0.5]], dtype=np.float32)
    loc = np.zeros((2, 2), dtype=np.float32)
    scale = np.ones((2, 2), dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 3: float64, 3D arrays
    x = np.random.uniform(0.1, 0.9, size=(2, 2, 2)).astype(np.float64)
    a = np.random.uniform(0.5, 5.0, size=(2, 2, 2)).astype(np.float64)
    b = np.random.uniform(0.5, 5.0, size=(2, 2, 2)).astype(np.float64)
    loc = np.zeros((2, 2, 2), dtype=np.float64)
    scale = np.ones((2, 2, 2), dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 4: Broadcasting, 1D with 0D-like (1-element) tensors
    x = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    a = np.array([2.0], dtype=np.float32)
    b = np.array([1.5], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 5: Custom loc and scale (loc = 1.0, scale = 3.0), support [1.0, 4.0]
    x = np.array([1.5, 2.0, 3.5], dtype=np.float32)
    a = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scale = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 6: Negative loc (loc = -5.0, scale = 10.0), support [-5.0, 5.0]
    x = np.array([-4.0, 0.0, 4.0], dtype=np.float32)
    a = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    loc = np.array([-5.0, -5.0, -5.0], dtype=np.float32)
    scale = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 7: High dimensions (4D)
    x = np.random.uniform(0.01, 0.99, size=(2, 2, 2, 2)).astype(np.float32)
    a = np.random.uniform(0.1, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    b = np.random.uniform(0.1, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 8: Boundary inputs (x near or at boundaries)
    x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    a = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.zeros(3, dtype=np.float32)
    scale = np.ones(3, dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 9: Small positive shape parameters (a < 1, b < 1)
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    b = np.array([0.3, 0.2, 0.1], dtype=np.float32)
    loc = np.zeros(3, dtype=np.float32)
    scale = np.ones(3, dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 10: Mixed broadcasting shapes
    x = np.array([[0.2, 0.5, 0.8]], dtype=np.float32)
    a = np.array([[2.0], [3.0]], dtype=np.float32)
    b = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    loc = np.array([[0.0]], dtype=np.float32)
    scale = np.array([[1.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.beta.logpdf"] = beta_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.beta.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.beta.logpdf'.")


check_valid('jax.scipy.stats.beta.logpdf', generated_inputs['jax.scipy.stats.beta.logpdf'], lib="jax", suffix=0)
