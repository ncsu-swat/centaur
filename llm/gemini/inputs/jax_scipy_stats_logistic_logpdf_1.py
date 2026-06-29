
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_logistic_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays, positive scale
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 arrays
    x = np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.5], [-0.5, 0.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 precision
    x = np.array([1.5, -2.5], dtype=np.float64)
    loc = np.array([-1.0, 2.0], dtype=np.float64)
    scale = np.array([0.5, 3.0], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar values represented as 0-D numpy arrays
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(-1.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broad-casting shapes
    x = np.random.randn(2, 3).astype(np.float32)
    loc = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    scale = np.array([[1.0], [2.0]], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large 1D array
    x = np.linspace(-10, 10, 100).astype(np.float32)
    loc = np.zeros(100, dtype=np.float32)
    scale = np.ones(100, dtype=np.float32) * 0.5
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: High-dimensional 3D array
    x = np.random.randn(2, 2, 2).astype(np.float32)
    loc = np.random.randn(2, 2, 2).astype(np.float32)
    scale = np.abs(np.random.randn(2, 2, 2)).astype(np.float32) + 0.1
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small scale values (float32)
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([0.01, 0.05, 0.1], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large scale values (float64)
    x = np.array([-100.0, 0.0, 100.0], dtype=np.float64)
    loc = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    scale = np.array([50.0, 100.0, 200.0], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-standard shapes and values
    x = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.2], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.logpdf_1"] = generate_logistic_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.logpdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.logpdf_1'.")


check_valid('jax.scipy.stats.logistic.logpdf', generated_inputs['jax.scipy.stats.logistic.logpdf_1'], lib="jax", suffix=1)
