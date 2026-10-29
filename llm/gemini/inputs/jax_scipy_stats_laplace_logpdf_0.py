
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def laplace_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard Laplace
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays, float32, positive values
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 1D arrays with negative x and loc
    x = np.array([-2.5, -1.0, 0.0, 1.5], dtype=np.float64)
    loc = np.array([-1.0, -1.0, 0.0, 0.5], dtype=np.float64)
    scale = np.array([0.5, 1.0, 2.0, 1.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: 2D arrays, matching shapes
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    scale = np.array([[1.0, 0.5], [2.0, 1.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting - 2D x with 1D loc and scalar scale
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: Broadcasting - 3D x, 2D loc, 1D scale
    x = np.random.randn(2, 3, 4).astype(np.float64)
    loc = np.zeros((3, 4), dtype=np.float64)
    scale = np.ones((4,), dtype=np.float64) * 0.8
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: 4D tensor with small scale values
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.full((2, 2, 2, 2), 0.1, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Large scale values
    x = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([50.0, 100.0, 200.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Integer arrays promoted to float
    x = np.array([-3, -1, 0, 1, 3], dtype=np.int32)
    loc = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    scale = np.array([1, 2, 1, 2, 1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: 2D arrays with large negative and positive values in float64
    x = np.array([[-1e5, 1e5], [0.0, -50.0]], dtype=np.float64)
    loc = np.array([[0.0, 10.0], [-10.0, 0.0]], dtype=np.float64)
    scale = np.array([[1e4, 1e4], [10.0, 5.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 11: Single element 1D arrays
    x = np.array([-0.75], dtype=np.float32)
    loc = np.array([0.25], dtype=np.float32)
    scale = np.array([1.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.laplace.logpdf"] = laplace_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.laplace.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.laplace.logpdf'.")


check_valid('jax.scipy.stats.laplace.logpdf', generated_inputs['jax.scipy.stats.laplace.logpdf'], lib="jax", suffix=0)
