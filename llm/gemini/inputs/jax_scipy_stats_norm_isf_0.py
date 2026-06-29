
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_norm_isf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, standard float32
    q = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D arrays (scalars as tensors), float64
    q = np.array(0.25, dtype=np.float64)
    loc = np.array(-1.0, dtype=np.float64)
    scale = np.array(2.5, dtype=np.float64)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, different loc and scales
    q = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    loc = np.array([[1.0, 2.0], [-1.0, -2.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting shapes: q is (3, 1), loc is (1, 4), scale is (1, 1)
    q = np.array([[0.05], [0.5], [0.95]], dtype=np.float32)
    loc = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    scale = np.array([[1.0]], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large dimension (3D)
    q = np.random.uniform(0.01, 0.99, size=(2, 3, 4)).astype(np.float32)
    loc = np.random.normal(0.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    scale = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Edge cases for q (very close to 0 and 1)
    q = np.array([1e-7, 0.5, 1.0 - 1e-7], dtype=np.float64)
    loc = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    scale = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Uniformly spaced q with matching loc and scale
    q = np.linspace(0.01, 0.99, 10, dtype=np.float32)
    loc = np.zeros(10, dtype=np.float32)
    scale = np.ones(10, dtype=np.float32) * 5.0
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small scale values (near zero standard deviation)
    q = np.array([0.2, 0.8], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1e-3, 1e-4], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large scale values
    q = np.array([0.3, 0.7], dtype=np.float64)
    loc = np.array([-100.0, 100.0], dtype=np.float64)
    scale = np.array([1000.0, 5000.0], dtype=np.float64)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D arrays
    q = np.random.uniform(0.1, 0.9, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.random.normal(0.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    scale = np.random.uniform(1.0, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.norm.isf"] = jax_scipy_stats_norm_isf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.isf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.isf'.")


check_valid('jax.scipy.stats.norm.isf', generated_inputs['jax.scipy.stats.norm.isf'], lib="jax", suffix=0)
