
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_ppf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays
    q = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 2: 1D arrays with matching shapes
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    loc = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    scale = np.array([1.0, 0.5, 2.0, 1.5, 0.8], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 3: 2D arrays float64
    q = np.random.uniform(0.01, 0.99, size=(3, 4)).astype(np.float64)
    loc = np.random.randn(3, 4).astype(np.float64)
    scale = np.random.uniform(0.5, 3.0, size=(3, 4)).astype(np.float64)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 4: 3D arrays float32
    q = np.random.uniform(0.05, 0.95, size=(2, 3, 4)).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 5: Broadcast shapes (3, 1) and (1, 4)
    q = np.linspace(0.1, 0.9, 3, dtype=np.float32).reshape(3, 1)
    loc = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 6: Boundary values close to 0 and 1
    q = np.array([1e-5, 1e-3, 0.5, 1 - 1e-3, 1 - 1e-5], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 7: Broadcasting 1D q with 1-element loc and scale
    q = np.linspace(0.01, 0.99, 10, dtype=np.float32)
    loc = np.array([5.0], dtype=np.float32)
    scale = np.array([2.5], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 8: 4D arrays float32
    q = np.random.uniform(0.1, 0.9, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.random.uniform(-5.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    scale = np.random.uniform(0.1, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 9: Negative loc and fractional scale float64
    q = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    loc = np.array([[-10.0, -20.0], [-30.0, -40.0]], dtype=np.float64)
    scale = np.array([[0.01, 0.05], [0.1, 0.2]], dtype=np.float64)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Input 10: Multi-dimensional broadcasting with differing rank arrays
    q = np.random.uniform(0.1, 0.9, size=(4, 1, 3)).astype(np.float32)
    loc = np.random.randn(1, 5, 1).astype(np.float32)
    scale = np.array([[[1.5]]], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    return [copy.deepcopy(x) for x in list_of_inputs]


generated_inputs["jax.scipy.stats.norm.ppf_1"] = norm_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.ppf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.ppf_1'.")


check_valid('jax.scipy.stats.norm.ppf', generated_inputs['jax.scipy.stats.norm.ppf_1'], lib="jax", suffix=1)
