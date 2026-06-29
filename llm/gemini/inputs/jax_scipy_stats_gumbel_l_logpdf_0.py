
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_l_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, standard scale
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
    scale = np.array([1.0, 1.5, 0.5, 2.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D arrays (scalars as tensors)
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.2, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float32, randomized values
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.random.randn(3, 4).astype(np.float32)
    scale = (np.abs(np.random.randn(3, 4)) + 0.1).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D arrays, float64
    x = np.random.randn(2, 2, 2).astype(np.float64)
    loc = np.random.randn(2, 2, 2).astype(np.float64)
    scale = (np.random.exponential(size=(2, 2, 2)) + 0.5).astype(np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting, x is 2D, loc/scale are 1D
    x = np.random.randn(5, 3).astype(np.float32)
    loc = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    scale = np.array([0.5, 1.2, 2.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting, x is 3D, loc/scale are 2D
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.random.randn(3, 4).astype(np.float32)
    scale = (np.random.rand(3, 4) + 0.1).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large negative values for x and loc
    x = np.array([-100.0, -50.0, -20.0], dtype=np.float32)
    loc = np.array([-80.0, -80.0, -80.0], dtype=np.float32)
    scale = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 high precision with large scales
    x = np.array([10.0, 20.0], dtype=np.float64)
    loc = np.array([5.0, 5.0], dtype=np.float64)
    scale = np.array([100.0, 200.0], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D arrays with small values
    x = (np.random.rand(2, 2, 2, 2) * 1e-3).astype(np.float32)
    loc = (np.random.rand(2, 2, 2, 2) * 1e-3).astype(np.float32)
    scale = (np.random.rand(2, 2, 2, 2) + 0.1).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D arrays with small positive scale
    x = np.array([0.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1e-4, 1e-4], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_l.logpdf"] = gumbel_l_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.logpdf'.")


check_valid('jax.scipy.stats.gumbel_l.logpdf', generated_inputs['jax.scipy.stats.gumbel_l.logpdf'], lib="jax", suffix=0)
