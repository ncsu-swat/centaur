
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gamma_sf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    x = np.array(1.5, dtype=np.float32)
    a = np.array(2.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays, same shape, standard values
    x = np.array([0.5, 1.0, 2.5, 5.0], dtype=np.float32)
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 3: Float64 precision, non-zero loc and scale
    x = np.array([2.0, 3.5, 5.0], dtype=np.float64)
    a = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    loc = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    scale = np.array([0.5, 2.0, 1.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 4: 2D arrays
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    loc = np.array([[0.0, 0.5], [-0.5, 1.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting with 1D and 2D arrays
    x = np.array([[0.5, 1.5, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)
    a = np.array([[1.0], [2.0]], dtype=np.float32)
    loc = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 6: Values of x less than loc (survival function should evaluate to 1.0)
    x = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
    a = np.array([2.0, 3.0, 1.0], dtype=np.float32)
    loc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 7: 3D tensors
    x = np.full((2, 3, 4), 3.0, dtype=np.float32)
    a = np.full((2, 3, 4), 2.5, dtype=np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 8: Small shape parameter (a < 1)
    x = np.array([0.01, 0.1, 0.5, 1.0], dtype=np.float64)
    a = np.array([0.1, 0.2, 0.5, 0.8], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 9: Large shape and scale parameters
    x = np.array([50.0, 100.0, 200.0], dtype=np.float32)
    a = np.array([20.0, 30.0, 50.0], dtype=np.float32)
    loc = np.array([5.0, 10.0, 0.0], dtype=np.float32)
    scale = np.array([5.0, 4.0, 10.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 10: 4D tensors with broadcasting
    x = np.random.uniform(1.0, 5.0, size=(2, 1, 3, 4)).astype(np.float32)
    a = np.random.uniform(0.5, 3.0, size=(1, 2, 1, 4)).astype(np.float32)
    loc = np.zeros((1, 1, 1, 1), dtype=np.float32)
    scale = np.random.uniform(0.5, 2.0, size=(2, 2, 3, 1)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.gamma.sf"] = gamma_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gamma.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gamma.sf'.")


check_valid('jax.scipy.stats.gamma.sf', generated_inputs['jax.scipy.stats.gamma.sf'], lib="jax", suffix=0)
