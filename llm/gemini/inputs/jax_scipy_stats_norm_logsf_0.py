
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_logsf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, standard normal values
    x = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: Scalar 0D arrays, float64
    x = np.array(1.5, dtype=np.float64)
    loc = np.array(-0.5, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays, matching shapes
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.zeros((3, 4), dtype=np.float32)
    scale = np.ones((3, 4), dtype=np.float32) * 1.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting - x is 2D, loc and scale are 1D
    x = np.linspace(-3.0, 3.0, 12, dtype=np.float32).reshape(3, 4)
    loc = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    scale = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Large positive and negative values (tail behavior)
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    loc = np.array([0.0], dtype=np.float64)
    scale = np.array([1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: 3D arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.abs(np.random.randn(2, 3, 4)).astype(np.float32) + 0.1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Broadcasting with shape (4, 1) and (1, 3)
    x = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    loc = np.array([[0.0, -1.0, 1.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Integer tensors
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    loc = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    scale = np.array([1, 2, 1, 2, 1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Non-standard scale and loc parameters with 1D inputs
    x = np.array([100.0, 105.0, 110.0], dtype=np.float64)
    loc = np.array([100.0, 100.0, 100.0], dtype=np.float64)
    scale = np.array([15.0, 15.0, 15.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: 4D tensor inputs
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2, 2), dtype=np.float32) * 2.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 11: Single element 1D arrays
    x = np.array([2.5], dtype=np.float32)
    loc = np.array([-1.0], dtype=np.float32)
    scale = np.array([0.1], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.norm.logsf"] = norm_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.logsf'.")


check_valid('jax.scipy.stats.norm.logsf', generated_inputs['jax.scipy.stats.norm.logsf'], lib="jax", suffix=0)
