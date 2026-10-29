
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_logsf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, standard cauchy
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 2: 1D arrays, float32
    x = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 3: 2D arrays, float64
    x = np.array([[-3.0, -1.5], [0.5, 2.0]], dtype=np.float64)
    loc = np.array([[1.0, -1.0], [0.0, 2.0]], dtype=np.float64)
    scale = np.array([[0.5, 1.5], [2.0, 0.8]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 4: Broadcasting (x is 2D, loc is 1D, scale is 0D)
    x = np.linspace(-5.0, 5.0, 12, dtype=np.float32).reshape(3, 4)
    loc = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 5: 3D arrays, float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 6: Large positive values of x (tail behavior)
    x = np.array([1e2, 1e4, 1e6], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 7: Large negative values of x
    x = np.array([-1e6, -1e4, -1e2], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 8: Small scale values
    x = np.array([0.0, 0.01, -0.01], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1e-3, 1e-3, 1e-3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 9: Non-zero negative locations
    x = np.array([[-10.0, -5.0], [0.0, 5.0]], dtype=np.float32)
    loc = np.array([[-15.0, -10.0], [-5.0, 0.0]], dtype=np.float32)
    scale = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 10: 4D array with broadcasting scale
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.random.uniform(-1.0, 1.0, size=(1, 2, 1, 2)).astype(np.float32)
    scale = np.random.uniform(0.5, 2.0, size=(2, 1, 2, 1)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.logsf_1"] = cauchy_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logsf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logsf_1'.")


check_valid('jax.scipy.stats.cauchy.logsf', generated_inputs['jax.scipy.stats.cauchy.logsf_1'], lib="jax", suffix=1)
