
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def truncnorm_cdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, standard bounds [-2, 2], loc=0, scale=1
    x = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    a = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 2: 0D scalars (tensors of shape ())
    x = np.array(0.5, dtype=np.float64)
    a = np.array(-1.0, dtype=np.float64)
    b = np.array(1.0, dtype=np.float64)
    loc = np.array(0.0, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays, float32
    x = np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    a = np.array([[-2.0, -2.0], [-2.0, -2.0]], dtype=np.float32)
    b = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting with different shapes
    x = np.linspace(-1.0, 1.0, 5, dtype=np.float32).reshape(5, 1)
    a = np.array([-3.0], dtype=np.float32)
    b = np.array([3.0], dtype=np.float32)
    loc = np.zeros((1, 3), dtype=np.float32)
    scale = np.ones((1, 1), dtype=np.float32) * 1.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 5: Strictly positive truncation bounds (one-sided-like)
    x = np.array([1.5, 2.0, 2.5], dtype=np.float64)
    a = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    b = np.array([3.0, 3.0, 3.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 6: Strictly negative truncation bounds
    x = np.array([-2.5, -2.0, -1.5], dtype=np.float32)
    a = np.array([-3.0, -3.0, -3.0], dtype=np.float32)
    b = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 7: Values of x outside the truncation bounds (evaluating support edges)
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float64)
    a = np.array([-2.0, -2.0, -2.0, -2.0, -2.0], dtype=np.float64)
    b = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 8: 3D arrays with non-trivial loc and scale
    x = np.random.uniform(1.0, 3.0, size=(2, 3, 4)).astype(np.float32)
    a = np.full((2, 3, 4), -1.0, dtype=np.float32)
    b = np.full((2, 3, 4), 2.0, dtype=np.float32)
    loc = np.full((2, 3, 4), 2.0, dtype=np.float32)
    scale = np.full((2, 3, 4), 0.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 9: Asymmetric truncation bounds with float64
    x = np.array([-0.5, 0.0, 1.0, 3.0], dtype=np.float64)
    a = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    b = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float64)
    loc = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    scale = np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    # Input 10: 4D batch inputs
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    a = np.full((2, 2, 2, 2), -2.5, dtype=np.float32)
    b = np.full((2, 2, 2, 2), 2.5, dtype=np.float32)
    loc = np.full((2, 2, 2, 2), -1.0, dtype=np.float32)
    scale = np.full((2, 2, 2, 2), 3.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.truncnorm.cdf"] = truncnorm_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.truncnorm.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.truncnorm.cdf'.")


check_valid('jax.scipy.stats.truncnorm.cdf', generated_inputs['jax.scipy.stats.truncnorm.cdf'], lib="jax", suffix=0)
