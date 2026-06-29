
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_l_logsf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D arrays (scalars as tensors), float32
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(-1.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float64
    x = np.array([[1.0, -2.0], [3.0, 0.5]], dtype=np.float64)
    loc = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float64)
    scale = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting shapes (x: 2D, loc: 1D, scale: scalar)
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.random.randn(4).astype(np.float32)
    scale = np.array([2.5], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative values for x and loc, float32
    x = np.array([-10.0, -20.0, -30.0], dtype=np.float32)
    loc = np.array([-5.0, -15.0, -25.0], dtype=np.float32)
    scale = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive values, small scale
    x = np.array([100.0, 200.0], dtype=np.float32)
    loc = np.array([50.0, 100.0], dtype=np.float32)
    scale = np.array([0.1, 0.2], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays, float32
    x = np.random.randn(2, 2, 2).astype(np.float32)
    loc = np.random.randn(2, 2, 2).astype(np.float32)
    scale = np.random.uniform(0.1, 5.0, size=(2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 broadcasting
    x = np.random.randn(5, 1, 3).astype(np.float64)
    loc = np.random.randn(1, 4, 3).astype(np.float64)
    scale = np.random.uniform(0.5, 2.0, size=(5, 4, 1)).astype(np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero values for x and loc, float32
    x = np.zeros((3, 3), dtype=np.float32)
    loc = np.zeros((3, 3), dtype=np.float32)
    scale = np.ones((3, 3), dtype=np.float32) * 2.0
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D arrays, float32
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    loc = np.random.randn(2, 2, 3, 3).astype(np.float32)
    scale = np.random.uniform(0.1, 10.0, size=(2, 2, 3, 3)).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_l.logsf"] = gumbel_l_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.logsf'.")


check_valid('jax.scipy.stats.gumbel_l.logsf', generated_inputs['jax.scipy.stats.gumbel_l.logsf'], lib="jax", suffix=0)
