
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_sf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 2D float64 arrays
    x = np.array([[-1.5, 0.5], [1.2, -0.8]], dtype=np.float64)
    loc = np.array([[0.0, 1.0], [-1.0, 0.5]], dtype=np.float64)
    scale = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: Broadcastable shapes (1D with 2D)
    x = np.array([[0.5, -0.5, 1.5]], dtype=np.float32)
    loc = np.array([[0.0]], dtype=np.float32)
    scale = np.array([[1.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: 3D float32 arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Scalar tensors (0D arrays)
    x = np.array(1.96, dtype=np.float64)
    loc = np.array(0.0, dtype=np.float64)
    scale = np.array(1.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: Large positive and negative values in x
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Varying scale values (strictly positive)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([0.1, 1.0, 10.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: 4D tensors with broadcasting
    x = np.random.randn(2, 1, 3, 4).astype(np.float32)
    loc = np.random.randn(1, 2, 3, 1).astype(np.float32)
    scale = np.abs(np.random.randn(1, 1, 1, 1)).astype(np.float32) + 0.1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Non-zero locations with float64 precision
    x = np.array([10.5, 20.2, -5.0, 100.0], dtype=np.float64)
    loc = np.array([10.0, 20.0, 0.0, 90.0], dtype=np.float64)
    scale = np.array([2.5, 5.0, 1.2, 10.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: Empty arrays (preserving dimension structure)
    x = np.empty((0, 5), dtype=np.float32)
    loc = np.empty((0, 5), dtype=np.float32)
    scale = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.norm.sf"] = norm_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.sf'.")


check_valid('jax.scipy.stats.norm.sf', generated_inputs['jax.scipy.stats.norm.sf'], lib="jax", suffix=0)
