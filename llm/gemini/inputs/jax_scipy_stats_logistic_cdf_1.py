
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logistic_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars
    x = np.array(0.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays with matching shapes
    x = np.array([-2.5, -1.0, 0.0, 1.5, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 2.0, 0.5, 1.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays with negative loc and varying positive scale
    x = np.array([[-1.0, 2.0], [0.5, -3.0]], dtype=np.float64)
    loc = np.array([[-2.0, 1.0], [0.0, -1.0]], dtype=np.float64)
    scale = np.array([[0.5, 1.2], [2.0, 0.8]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting 1D loc and scale with 2D x
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    scale = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting column vectors and row vectors
    x = np.linspace(-5.0, 5.0, 6, dtype=np.float32).reshape(6, 1)
    loc = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
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

    # Input 7: 4D tensors with float64
    x = np.random.randn(2, 2, 3, 3).astype(np.float64)
    loc = np.zeros((2, 2, 3, 3), dtype=np.float64)
    scale = np.ones((2, 2, 3, 3), dtype=np.float64) * 2.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Extreme values for x with scalar-like 1D arrays
    x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Non-zero loc and small scale
    x = np.array([10.0, 10.5, 11.0], dtype=np.float32)
    loc = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    scale = np.array([0.01, 0.05, 0.1], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: Broadcasting 3D with 1D
    x = np.random.randn(4, 3, 2).astype(np.float32)
    loc = np.array([0.5, -0.5], dtype=np.float32)
    scale = np.array([1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 11: 1D empty array
    x = np.array([], dtype=np.float32)
    loc = np.array([], dtype=np.float32)
    scale = np.array([], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.logistic.cdf_1"] = logistic_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.cdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.cdf_1'.")


check_valid('jax.scipy.stats.logistic.cdf', generated_inputs['jax.scipy.stats.logistic.cdf_1'], lib="jax", suffix=1)
