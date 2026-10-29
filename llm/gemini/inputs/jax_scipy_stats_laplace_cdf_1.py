
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def laplace_cdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays with float32
    x = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: Scalar (0-D) arrays
    x = np.array(0.5, dtype=np.float64)
    loc = np.array(-1.0, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays with identical shape
    x = np.array([[-2.0, 0.0], [1.5, 3.0]], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0], [2.0, 0.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting 1D with 2D
    x = np.zeros((3, 4), dtype=np.float32)
    loc = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scale = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Large positive and negative values in x (tail behavior)
    x = np.array([-100.0, -10.0, 0.0, 10.0, 100.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 2.0, 1.0, 0.5, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: 3D arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Broadcasting scalar-like shape (1,) to multi-dimensional array
    x = np.linspace(-5.0, 5.0, 10, dtype=np.float64)
    loc = np.array([0.0], dtype=np.float64)
    scale = np.array([1.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Varying scale parameters (small and large scales)
    x = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([0.01, 1.0, 100.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Negative loc values
    x = np.array([-5.0, -3.0, -1.0], dtype=np.float32)
    loc = np.array([-4.0, -4.0, -4.0], dtype=np.float32)
    scale = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: 4D tensor inputs
    x = np.ones((1, 2, 2, 3), dtype=np.float64)
    loc = np.zeros((1, 2, 2, 3), dtype=np.float64)
    scale = np.full((1, 2, 2, 3), 2.5, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 11: Empty array input
    x = np.array([], dtype=np.float32)
    loc = np.array([], dtype=np.float32)
    scale = np.array([], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.laplace.cdf_1"] = laplace_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.laplace.cdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.laplace.cdf_1'.")


check_valid('jax.scipy.stats.laplace.cdf', generated_inputs['jax.scipy.stats.laplace.cdf_1'], lib="jax", suffix=1)
