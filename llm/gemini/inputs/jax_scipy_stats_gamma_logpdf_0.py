
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gamma_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays (float32)
    x = np.array(2.5, dtype=np.float32)
    a = np.array(1.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays of matching shape (float64)
    x = np.array([0.5, 1.0, 2.0, 3.5], dtype=np.float64)
    a = np.array([2.0, 2.0, 3.0, 1.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float64)
    scale = np.array([1.0, 2.0, 1.5, 0.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays (float32)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    loc = np.zeros((2, 2), dtype=np.float32)
    scale = np.ones((2, 2), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 4: 3D arrays with positive shift (loc > 0)
    x = np.ones((2, 3, 4), dtype=np.float32) * 5.0
    a = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    loc = np.ones((2, 3, 4), dtype=np.float32) * 1.0
    scale = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 5: Negative loc (valid support shifted to left)
    x = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 6: Broadcasting 2D with 1D and scalar
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    a = np.array([1.5], dtype=np.float32)
    loc = np.array([[0.0], [1.0]], dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 7: Fractional shape parameter a < 1
    x = np.array([0.01, 0.05, 0.1, 0.5], dtype=np.float64)
    a = np.array([0.1, 0.2, 0.5, 0.8], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 8: Large shape and scale parameters
    x = np.array([100.0, 150.0, 200.0], dtype=np.float32)
    a = np.array([50.0, 60.0, 70.0], dtype=np.float32)
    loc = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    scale = np.array([2.0, 2.5, 3.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 9: 4D tensors
    x = np.full((1, 2, 2, 3), 3.0, dtype=np.float32)
    a = np.full((1, 2, 2, 3), 2.5, dtype=np.float32)
    loc = np.full((1, 2, 2, 3), 0.0, dtype=np.float32)
    scale = np.full((1, 2, 2, 3), 1.2, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    # Input 10: Broadcasting with column vector x and row vector a
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    a = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "a": a, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.gamma.logpdf"] = gamma_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gamma.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gamma.logpdf'.")


check_valid('jax.scipy.stats.gamma.logpdf', generated_inputs['jax.scipy.stats.gamma.logpdf'], lib="jax", suffix=0)
