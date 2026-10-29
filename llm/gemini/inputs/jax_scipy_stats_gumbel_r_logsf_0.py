
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_r_logsf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard float32
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays with float64
    x = np.array([-2.0, 0.0, 1.5, 3.0], dtype=np.float64)
    loc = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float64)
    scale = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays, matching shapes
    x = np.array([[-1.0, 2.0], [0.5, -0.5]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting scalar loc and scale with 1D x
    x = np.linspace(-5.0, 5.0, 10, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting (3, 1) and (1, 4) to (3, 4)
    x = np.zeros((3, 4), dtype=np.float32)
    loc = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: 3D arrays with strictly positive x and scale
    x = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    loc = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    scale = np.random.uniform(0.5, 3.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Float64 high precision with negative values of x
    x = np.array([-10.0, -5.0, -2.5], dtype=np.float64)
    loc = np.array([-5.0, -5.0, -5.0], dtype=np.float64)
    scale = np.array([0.1, 0.5, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: 4D tensors
    x = np.ones((2, 2, 2, 2), dtype=np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.full((2, 2, 2, 2), 2.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Large positive values of x
    x = np.array([10.0, 20.0, 50.0], dtype=np.float32)
    loc = np.array([0.0, 5.0, 10.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: 1D array broadcasted with 2D array
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    loc = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    scale = np.array([[1.0], [2.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.gumbel_r.logsf"] = gumbel_r_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.logsf'.")


check_valid('jax.scipy.stats.gumbel_r.logsf', generated_inputs['jax.scipy.stats.gumbel_r.logsf'], lib="jax", suffix=0)
