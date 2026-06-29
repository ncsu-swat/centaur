
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_logistic_isf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32, standard parameters
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D (Scalar) float32
    x = np.array(0.25, dtype=np.float32)
    loc = np.array(1.5, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 with positive/negative locs and varying scales
    x = np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.5], [1.0, 2.5]], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 inputs
    x = np.array([0.05, 0.95], dtype=np.float64)
    loc = np.array([10.0, -10.0], dtype=np.float64)
    scale = np.array([0.1, 0.1], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 randomly generated
    x = np.random.uniform(0.01, 0.99, size=(2, 2, 2)).astype(np.float32)
    loc = np.random.normal(0.0, 5.0, size=(2, 2, 2)).astype(np.float32)
    scale = np.random.uniform(0.1, 10.0, size=(2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting shapes
    x = np.array([[0.2], [0.5], [0.8]], dtype=np.float32)  # shape (3, 1)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float32)     # shape (3,) -> broadcasts to (1, 3) -> (3, 3)
    scale = np.array([[1.0, 2.0, 3.0]], dtype=np.float32) # shape (1, 3)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Extreme probabilities close to 0 and 1
    x = np.array([1e-5, 0.5, 1.0 - 1e-5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large scales
    x = np.array([0.3, 0.7], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1000.0, 10000.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very small scales
    x = np.array([0.4, 0.6], dtype=np.float32)
    loc = np.array([-5.0, 5.0], dtype=np.float32)
    scale = np.array([1e-4, 1e-4], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional broadcasting with mismatched but compatible sizes
    x = np.array([0.5], dtype=np.float32)
    loc = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale = np.array([2.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.isf"] = jax_scipy_stats_logistic_isf_inputs()

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
