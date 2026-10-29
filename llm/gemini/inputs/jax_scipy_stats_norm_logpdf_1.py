
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays with matching shapes
    x = np.array([-1.5, 0.0, 1.5, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays, float64
    x = np.random.randn(3, 4).astype(np.float64)
    loc = np.zeros((3, 4), dtype=np.float64)
    scale = np.ones((3, 4), dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting loc and scale with 2D x
    x = np.random.randn(4, 5).astype(np.float32)
    loc = np.array([[1.0]], dtype=np.float32)
    scale = np.array([[2.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcast across axes (loc column, scale row)
    x = np.random.randn(5, 3).astype(np.float32)
    loc = np.array([[-1.0], [0.0], [1.0], [2.0], [-2.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: 3D tensors
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.abs(np.random.randn(2, 3, 4)).astype(np.float32) + 0.1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: 4D tensors
    x = np.random.randn(2, 2, 3, 3).astype(np.float64)
    loc = np.zeros((2, 2, 3, 3), dtype=np.float64)
    scale = np.full((2, 2, 3, 3), 1.5, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Negative loc and small positive scale
    x = np.array([-10.0, -5.0, 0.0, 5.0], dtype=np.float32)
    loc = np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float32)
    scale = np.array([0.01, 0.1, 1.0, 10.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Large dimensions
    x = np.random.randn(10, 20).astype(np.float32)
    loc = np.random.randn(10, 20).astype(np.float32)
    scale = np.random.uniform(0.5, 5.0, size=(10, 20)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: 1D broadcasting with 0D params
    x = np.linspace(-3.0, 3.0, 100, dtype=np.float64)
    loc = np.array(0.0, dtype=np.float64)
    scale = np.array(1.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.norm.logpdf_1"] = norm_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.logpdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.logpdf_1'.")


check_valid('jax.scipy.stats.norm.logpdf', generated_inputs['jax.scipy.stats.norm.logpdf_1'], lib="jax", suffix=1)
