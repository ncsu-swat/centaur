
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def uniform_ppf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, standard uniform [0, 1]
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "loc": loc, "scale": scale}))

    # Input 2: 0D scalar arrays
    q = np.array(0.42, dtype=np.float64)
    loc = np.array(2.0, dtype=np.float64)
    scale = np.array(5.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"q": q, "loc": loc, "scale": scale}))

    # Input 3: 2D arrays with identical shapes
    q = np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)
    loc = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    scale = np.array([[2.0, 3.0], [1.5, 0.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "loc": loc, "scale": scale}))

    # Input 4: Broadcasting with 2D and 1D arrays
    q = np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(2, 3)
    loc = np.array([[-5.0]], dtype=np.float32)
    scale = np.array([10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "loc": loc, "scale": scale}))

    # Input 5: 3D arrays
    q = np.random.uniform(0.0, 1.0, size=(2, 3, 4)).astype(np.float64)
    loc = np.zeros((2, 3, 4), dtype=np.float64)
    scale = np.ones((2, 3, 4), dtype=np.float64) * 2.5
    list_of_inputs.append(copy.deepcopy({"q": q, "loc": loc, "scale": scale}))

    # Input 6: Boundary values q=0.0 and q=1.0
    q = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    loc = np.array([10.0, -10.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 20.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "loc": loc, "scale": scale}))

    # Input 7: Broadcasting column and row vectors
    q = np.array([[0.2], [0.4], [0.6]], dtype=np.float32)
    loc = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "loc": loc, "scale": scale}))

    # Input 8: 4D tensor inputs
    q = np.random.uniform(0.1, 0.9, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.full((2, 2, 2, 2), -2.0, dtype=np.float32)
    scale = np.full((2, 2, 2, 2), 4.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "loc": loc, "scale": scale}))

    # Input 9: Negative loc values and fractional scale
    q = np.array([0.05, 0.15, 0.85, 0.95], dtype=np.float64)
    loc = np.array([-100.0, -50.0, -20.0, -10.0], dtype=np.float64)
    scale = np.array([0.01, 0.1, 0.5, 0.25], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"q": q, "loc": loc, "scale": scale}))

    # Input 10: Empty arrays
    q = np.array([], dtype=np.float32)
    loc = np.array([], dtype=np.float32)
    scale = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"q": q, "loc": loc, "scale": scale}))

    # Input 11: Single element 1D arrays with large scale
    q = np.array([0.73], dtype=np.float64)
    loc = np.array([1e5], dtype=np.float64)
    scale = np.array([1e6], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"q": q, "loc": loc, "scale": scale}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.uniform.ppf"] = uniform_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.uniform.ppf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.uniform.ppf'.")


check_valid('jax.scipy.stats.uniform.ppf', generated_inputs['jax.scipy.stats.uniform.ppf'], lib="jax", suffix=0)
