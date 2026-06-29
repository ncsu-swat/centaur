
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logsf_inputs():
    list_of_inputs = []

    # Input 1: Scalar-like (0D arrays)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, float32, standard normal parameters
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float64, non-standard loc and scale
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    loc = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    scale = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive x values (where logsf will be very negative)
    x = np.array([10.0, 20.0, 50.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative x values (where logsf will be close to 0)
    x = np.array([-10.0, -20.0, -50.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting with 2D x, 1D loc, 0D scale
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting with 1D x, 2D loc, 2D scale
    x = np.array([1.0, 2.0], dtype=np.float32)
    loc = np.random.randn(3, 2).astype(np.float32)
    scale = np.abs(np.random.randn(3, 2)).astype(np.float32) + 0.1
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays with large scale (flat distribution)
    x = np.random.randn(2, 2, 2).astype(np.float32)
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32) * 10.0
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small scale values (narrow distribution)
    x = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([0.01, 0.01, 0.01], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D arrays, float64
    x = np.random.randn(2, 2, 2, 2).astype(np.float64)
    loc = np.random.randn(2, 2, 2, 2).astype(np.float64)
    scale = np.abs(np.random.randn(2, 2, 2, 2)).astype(np.float64) + 0.5
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.norm.logsf"] = logsf_inputs()

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
