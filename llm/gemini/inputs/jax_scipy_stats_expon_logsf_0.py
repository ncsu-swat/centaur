
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expon_logsf_inputs():
    list_of_inputs = []

    # Input 1, valid — 0D scalar-like arrays
    x = np.array(2.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid — 1D arrays
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid — 2D arrays with float64
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    loc = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    scale = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid — Broadcasting
    x = np.random.uniform(1.0, 5.0, size=(3, 4)).astype(np.float32)
    loc = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid — Negative values of x
    x = np.array([-1.0, -2.0, 0.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid — Large scale values
    x = np.array([10.0, 20.0], dtype=np.float32)
    loc = np.array([1.0, 1.0], dtype=np.float32)
    scale = np.array([100.0, 200.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid — Small scale values
    x = np.array([0.1, 0.2], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([0.01, 0.02], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid — 3D arrays
    x = np.random.uniform(5.0, 10.0, size=(2, 2, 2)).astype(np.float64)
    loc = np.random.uniform(0.0, 2.0, size=(2, 2, 2)).astype(np.float64)
    scale = np.random.uniform(1.0, 3.0, size=(2, 2, 2)).astype(np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid — Mixed types (integers and floats)
    x = np.array([5, 6, 7], dtype=np.int32)
    loc = np.array([1, 2, 3], dtype=np.int32)
    scale = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid — 4D arrays
    x = np.ones((2, 3, 4, 5), dtype=np.float32) * 5.0
    loc = np.ones((2, 3, 4, 5), dtype=np.float32) * 2.0
    scale = np.ones((2, 3, 4, 5), dtype=np.float32) * 3.0
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11, valid — Another broadcasting case
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([[1.0], [2.0]], dtype=np.float32)
    scale = np.array([0.5], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.expon.logsf"] = expon_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.logsf'.")


check_valid('jax.scipy.stats.expon.logsf', generated_inputs['jax.scipy.stats.expon.logsf'], lib="jax", suffix=0)
