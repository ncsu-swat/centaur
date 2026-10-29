
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def truncnorm_sf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32, standard range
    x = np.array(0.5, dtype=np.float32)
    a = np.array(-1.0, dtype=np.float32)
    b = np.array(2.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays, same shapes, float64
    x = np.array([-1.5, 0.0, 1.2], dtype=np.float64)
    a = np.array([-2.0, -1.0, 0.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    loc = np.array([0.0, 0.5, -0.5], dtype=np.float64)
    scale = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays, shape (2, 2)
    x = np.array([[0.2, 0.8], [-0.5, 1.5]], dtype=np.float32)
    a = np.array([[-1.0, -0.5], [-2.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [0.5, 3.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting: x is (3, 1), bounds are (1, 4), loc/scale are scalars
    x = np.array([[-0.5], [0.0], [0.5]], dtype=np.float32)
    a = np.array([[-2.0, -1.5, -1.0, -0.5]], dtype=np.float32)
    b = np.array([[1.0, 1.5, 2.0, 2.5]], dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 5: x outside bounds (lower than loc + a * scale)
    x = np.array([-10.0, -5.0], dtype=np.float32)
    a = np.array([-1.0, -2.0], dtype=np.float32)
    b = np.array([2.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 1.0], dtype=np.float32)
    scale = np.array([1.0, 0.8], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 6: x outside bounds (greater than loc + b * scale)
    x = np.array([10.0, 5.0], dtype=np.float32)
    a = np.array([-1.0, -2.0], dtype=np.float32)
    b = np.array([2.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 1.0], dtype=np.float32)
    scale = np.array([1.0, 0.8], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 7: 3D tensor
    x = np.zeros((2, 2, 2), dtype=np.float32)
    a = np.full((2, 2, 2), -2.5, dtype=np.float32)
    b = np.full((2, 2, 2), 2.5, dtype=np.float32)
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 8: Strictly positive support, float64
    x = np.array([1.5, 2.0, 3.5], dtype=np.float64)
    a = np.array([1.0, 0.5, 2.0], dtype=np.float64)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 9: Strictly negative support
    x = np.array([-3.5, -2.0, -1.5], dtype=np.float32)
    a = np.array([-5.0, -4.0, -3.0], dtype=np.float32)
    b = np.array([-1.0, -0.5, -0.1], dtype=np.float32)
    loc = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    scale = np.array([1.2, 0.9, 1.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 10: Non-trivial location and scale parameters with broadcast
    x = np.array([[5.0, 10.0, 15.0]], dtype=np.float32)
    a = np.array([[-3.0], [-2.0]], dtype=np.float32)
    b = np.array([[3.0], [2.0]], dtype=np.float32)
    loc = np.array([[10.0]], dtype=np.float32)
    scale = np.array([[2.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

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
