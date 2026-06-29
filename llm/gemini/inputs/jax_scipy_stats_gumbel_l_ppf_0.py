
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_l_ppf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    p = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 2: 2D float32 arrays
    p = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    loc = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 3: 1D float64 arrays with wider bounds
    p = np.array([0.01, 0.99], dtype=np.float64)
    loc = np.array([-10.0, 10.0], dtype=np.float64)
    scale = np.array([0.1, 5.0], dtype=np.float64)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 4: Broadcastable shapes (5, 1), (1, 5), (5, 5)
    p = np.random.uniform(0.1, 0.9, size=(5, 1)).astype(np.float32)
    loc = np.random.uniform(-1, 1, size=(1, 5)).astype(np.float32)
    scale = np.random.uniform(0.1, 2.0, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 5: 3D float32 arrays
    p = np.random.uniform(0.1, 0.9, size=(2, 3, 4)).astype(np.float32)
    loc = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    scale = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 6: Large scale factor
    p = np.array([0.5], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([100.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 7: Small scale factor with negative loc
    p = np.array([0.5], dtype=np.float32)
    loc = np.array([-50.0], dtype=np.float32)
    scale = np.array([0.01], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 8: 4D float32 arrays
    p = np.random.uniform(0.05, 0.95, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 9: Single-element float64 arrays
    p = np.array([0.5], dtype=np.float64)
    loc = np.array([1.5], dtype=np.float64)
    scale = np.array([2.5], dtype=np.float64)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 10: Randomized 1D arrays
    p = np.random.uniform(0.1, 0.9, size=(10,)).astype(np.float32)
    loc = np.random.normal(0, 1, size=(10,)).astype(np.float32)
    scale = np.random.exponential(1.0, size=(10,)).astype(np.float32) + 0.1
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_l.ppf"] = gumbel_l_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.ppf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.ppf'.")


check_valid('jax.scipy.stats.gumbel_l.ppf', generated_inputs['jax.scipy.stats.gumbel_l.ppf'], lib="jax", suffix=0)
