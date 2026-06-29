
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_uniform_pdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, standard range
    x = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float64
    x = np.array([[0.1, 0.2], [0.8, 1.2]], dtype=np.float64)
    loc = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative loc, large scale
    x = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    loc = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
    scale = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting 1D x with 0D loc and scale (represented as 0-dim arrays)
    x = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    loc = np.array(0.1, dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays, float32
    x = np.random.uniform(0, 5, size=(2, 2, 2)).astype(np.float32)
    loc = np.random.uniform(-1, 1, size=(2, 2, 2)).astype(np.float32)
    scale = np.random.uniform(1, 5, size=(2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer types
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    loc = np.array([0, 1, 2, 3], dtype=np.int32)
    scale = np.array([2, 2, 2, 2], dtype=np.int32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very small scale
    x = np.array([1.001, 1.002, 1.005], dtype=np.float32)
    loc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scale = np.array([0.01, 0.01, 0.01], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D arrays, float64
    x = np.random.uniform(-2, 2, size=(2, 2, 1, 3)).astype(np.float64)
    loc = np.zeros((2, 2, 1, 3), dtype=np.float64)
    scale = np.ones((2, 2, 1, 3), dtype=np.float64) * 2.0
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed dimensions broadcasting (x is 2x3, loc is 1x3, scale is 2x1)
    x = np.array([[0.5, 1.5, 2.5], [1.0, 2.0, 3.0]], dtype=np.float32)
    loc = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    scale = np.array([[1.0], [2.0]], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scalar arrays (0D tensors)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D arrays with wider range
    x = np.linspace(-1, 2, 10, dtype=np.float32)
    loc = np.array([-0.5], dtype=np.float32)
    scale = np.array([1.5], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.uniform.pdf"] = jax_scipy_stats_uniform_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.uniform.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.uniform.pdf'.")


check_valid('jax.scipy.stats.uniform.pdf', generated_inputs['jax.scipy.stats.uniform.pdf'], lib="jax", suffix=0)
