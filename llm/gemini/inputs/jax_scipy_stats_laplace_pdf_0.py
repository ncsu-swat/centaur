
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def laplace_pdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float32
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.random.randn(3, 4).astype(np.float32)
    scale = np.abs(np.random.randn(3, 4)).astype(np.float32) + 0.1
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D arrays, float64
    x = np.random.randn(2, 2, 2).astype(np.float64)
    loc = np.random.randn(2, 2, 2).astype(np.float64)
    scale = np.random.uniform(0.1, 5.0, size=(2, 2, 2)).astype(np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting, x 2D, loc 1D, scale 1D
    x = np.random.randn(4, 5).astype(np.float32)
    loc = np.array([0.5, -0.5, 0.0, 1.2], dtype=np.float32).reshape(4, 1)
    scale = np.array([1.0, 2.0, 1.5, 0.8], dtype=np.float32).reshape(4, 1)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar tensors (0D arrays)
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(-0.5, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large scale values
    x = np.random.randn(5).astype(np.float32)
    loc = np.zeros(5, dtype=np.float32)
    scale = np.array([10.0, 50.0, 100.0, 500.0, 1000.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small scale values (must be positive, > 0)
    x = np.random.randn(5).astype(np.float32)
    loc = np.zeros(5, dtype=np.float32)
    scale = np.array([0.1, 0.05, 0.01, 0.005, 0.001], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative loc, positive x
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional array (4D)
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2, 2), dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting scale as scalar but representation as 1-element array
    x = np.random.randn(3, 3).astype(np.float64)
    loc = np.random.randn(3, 3).astype(np.float64)
    scale = np.array([2.5], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.laplace.pdf"] = laplace_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.laplace.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.laplace.pdf'.")


check_valid('jax.scipy.stats.laplace.pdf', generated_inputs['jax.scipy.stats.laplace.pdf'], lib="jax", suffix=0)
