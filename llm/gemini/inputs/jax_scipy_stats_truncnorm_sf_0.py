
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def truncnorm_sf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32, symmetric bounds
    x = np.array([0.0, -0.5, 0.5, 1.0, -1.0], dtype=np.float32)
    a = np.array([-2.0, -2.0, -2.0, -2.0, -2.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float32, positive truncation bounds
    x = np.array([[1.5, 2.0], [2.5, 3.0]], dtype=np.float32)
    a = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    b = np.array([[4.0, 4.0], [4.0, 4.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64, negative truncation bounds
    x = np.array([-2.5, -2.0, -1.5], dtype=np.float64)
    a = np.array([-3.0, -3.0, -3.0], dtype=np.float64)
    b = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar-like (0D arrays)
    x = np.array(0.5, dtype=np.float32)
    a = np.array(-1.0, dtype=np.float32)
    b = np.array(1.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large scale and non-zero loc
    x = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    a = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    scale = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broad-casted dimensions (x is 2D, params are 1D)
    x = np.random.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
    a = np.array([-2.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: High dimensions (3D)
    x = np.random.uniform(0.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    a = np.full((2, 2, 2), -1.5, dtype=np.float32)
    b = np.full((2, 2, 2), 1.5, dtype=np.float32)
    loc = np.full((2, 2, 2), 0.5, dtype=np.float32)
    scale = np.full((2, 2, 2), 0.5, dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x values outside the support [loc + a*scale, loc + b*scale]
    x = np.array([-3.0, 3.0, 0.0], dtype=np.float32)
    a = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small positive scales and tight truncation
    x = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Asymmetric varying limits per element
    x = np.array([-0.5, 0.5, 1.5], dtype=np.float64)
    a = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    loc = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    scale = np.array([0.8, 1.2, 1.5], dtype=np.float64)
    input_dict = {"x": x, "a": a, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.truncnorm.sf"] = truncnorm_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.truncnorm.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.truncnorm.sf'.")


check_valid('jax.scipy.stats.truncnorm.sf', generated_inputs['jax.scipy.stats.truncnorm.sf'], lib="jax", suffix=0)
