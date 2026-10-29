
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, float32
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays, float32, standard values
    x = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays, float64
    x = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float64)
    loc = np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float64)
    scale = np.array([[0.5, 1.5], [2.0, 0.1]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: 3D arrays, float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting - 2D x with column loc and row scale
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: Broadcasting - 3D x with 0D loc and scale
    x = np.random.randn(2, 2, 2).astype(np.float64)
    loc = np.array(0.0, dtype=np.float64)
    scale = np.array(2.5, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Large magnitude x values (tails of Cauchy)
    x = np.array([-1e6, -100.0, 100.0, 1e6], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: 4D arrays, float32
    x = np.random.randn(2, 1, 3, 2).astype(np.float32)
    loc = np.random.randn(2, 1, 3, 2).astype(np.float32)
    scale = np.random.uniform(0.5, 2.0, size=(2, 1, 3, 2)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Non-zero location shifts with small scale
    x = np.array([10.0, 10.5, 11.0], dtype=np.float64)
    loc = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    scale = np.array([0.01, 0.05, 0.1], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: Broadcasting 1D array against 2D array
    x = np.linspace(-3.0, 3.0, 5, dtype=np.float32)
    loc = np.array([[0.0], [1.0]], dtype=np.float32)
    scale = np.array([[1.0], [0.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.cdf_1"] = cauchy_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.cdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.cdf_1'.")


check_valid('jax.scipy.stats.cauchy.cdf', generated_inputs['jax.scipy.stats.cauchy.cdf_1'], lib="jax", suffix=1)
