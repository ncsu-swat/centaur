
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_beta_pdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, standard beta (loc=0, scale=1)
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    a = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0-dimensional arrays (scalars as tensors)
    x = np.array(0.5, dtype=np.float32)
    a = np.array(1.5, dtype=np.float32)
    b = np.array(2.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 2D arrays
    x = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    a = np.array([[2.0, 2.5], [3.0, 3.5]], dtype=np.float64)
    b = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float64)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting shapes
    x = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)  # shape (3, 1)
    a = np.array([[1.0, 2.0]], dtype=np.float32)           # shape (1, 2)
    b = np.array([[2.0, 3.0]], dtype=np.float32)           # shape (1, 2)
    loc = np.array([0.0], dtype=np.float32)                # shape (1,)
    scale = np.array([1.0], dtype=np.float32)              # shape (1,)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Non-standard loc and scale
    # Standardized variable y = (x - loc) / scale must be in [0, 1]
    # For loc=1.0, scale=2.0, x must be in [1.0, 3.0]
    x = np.array([1.2, 2.0, 2.8], dtype=np.float32)
    a = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D random arrays
    x = np.random.uniform(0.1, 0.9, size=(2, 3, 2)).astype(np.float32)
    a = np.random.uniform(0.5, 5.0, size=(2, 3, 2)).astype(np.float32)
    b = np.random.uniform(0.5, 5.0, size=(2, 3, 2)).astype(np.float32)
    loc = np.zeros((2, 3, 2), dtype=np.float32)
    scale = np.ones((2, 3, 2), dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small parameters < 1.0
    x = np.array([0.01, 0.99], dtype=np.float32)
    a = np.array([0.1, 0.1], dtype=np.float32)
    b = np.array([0.2, 0.2], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative loc and larger scale
    # loc = -10, scale = 20. x is in [-10, 10]
    x = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    a = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    b = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
    scale = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Integer shape parameters
    x = np.array([0.25, 0.75], dtype=np.float32)
    a = np.array([2, 3], dtype=np.int32)
    b = np.array([4, 5], dtype=np.int32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D arrays, float64
    x = np.random.uniform(0.1, 0.9, size=(2, 2, 2, 2)).astype(np.float64)
    a = np.random.uniform(1.0, 10.0, size=(2, 2, 2, 2)).astype(np.float64)
    b = np.random.uniform(1.0, 10.0, size=(2, 2, 2, 2)).astype(np.float64)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float64)
    scale = np.ones((2, 2, 2, 2), dtype=np.float64)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.beta.pdf"] = jax_scipy_stats_beta_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.beta.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.beta.pdf'.")


check_valid('jax.scipy.stats.beta.pdf', generated_inputs['jax.scipy.stats.beta.pdf'], lib="jax", suffix=0)
