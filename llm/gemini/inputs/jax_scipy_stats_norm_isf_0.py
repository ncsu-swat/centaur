
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_isf_inputs():
    list_of_inputs = []

    # Input 1: 1D tensors with standard normal
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 2: Scalar (0D) tensors
    q = np.array(0.05, dtype=np.float32)
    loc = np.array(10.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 3: 2D tensors float64
    q = np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float64)
    loc = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    scale = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting 1D loc and scale with 2D q
    q = np.full((3, 4), 0.3, dtype=np.float32)
    loc = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    scale = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 5: 3D tensors
    q = np.random.uniform(0.01, 0.99, size=(2, 3, 4)).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 6: Boundary probabilities 0.0 and 1.0
    q = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 7: Negative loc and various scales
    q = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    loc = np.array([-100.0, -50.0, -10.0], dtype=np.float64)
    scale = np.array([0.01, 10.0, 100.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 8: 4D tensors
    q = np.random.uniform(0.1, 0.9, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.random.randn(2, 2, 2, 2).astype(np.float32)
    scale = np.abs(np.random.randn(2, 2, 2, 2)).astype(np.float32) + 0.1
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 9: Extreme tail probabilities
    q = np.array([1e-5, 1e-4, 1e-3, 1 - 1e-3], dtype=np.float64)
    loc = np.array([0.0], dtype=np.float64)
    scale = np.array([1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 10: Column vector broadcasting with row vectors
    q = np.array([[0.2], [0.4], [0.6]], dtype=np.float32)
    loc = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 1.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.norm.isf"] = norm_isf_inputs()

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
