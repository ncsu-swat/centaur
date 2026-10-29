
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def uniform_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, float32, x within [loc, loc + scale]
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays of matching shape, x below loc (cdf should be 0)
    x = np.array([-1.0, -0.5, -2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([2.0, 1.0, 3.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 1D arrays of matching shape, x above loc + scale (cdf should be 1)
    x = np.array([5.0, 10.0, 3.5], dtype=np.float32)
    loc = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    scale = np.array([2.0, 3.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: 2D arrays, float64, negative loc
    x = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    loc = np.array([[-2.0, -2.0], [-2.0, -2.0]], dtype=np.float64)
    scale = np.array([[4.0, 4.0], [4.0, 4.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting - 2D x with 1D loc and scalar scale
    x = np.random.uniform(0.0, 5.0, size=(4, 3)).astype(np.float32)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array(5.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: Broadcasting - 3D arrays with compatible shapes
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    loc = np.random.uniform(-10.0, -5.0, size=(1, 3, 1)).astype(np.float32)
    scale = np.random.uniform(10.0, 20.0, size=(2, 1, 4)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: 1D array with mixed values (below, within, and above range)
    x = np.array([-2.0, 0.0, 0.5, 1.0, 1.5, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Float64 high-precision 2D inputs
    x = np.linspace(-10, 10, 12, dtype=np.float64).reshape(3, 4)
    loc = np.full((3, 4), -5.0, dtype=np.float64)
    scale = np.full((3, 4), 10.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: 4D arrays, float32
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: All negative parameters for loc and negative x
    x = np.array([-15.0, -12.0, -8.0, -3.0], dtype=np.float32)
    loc = np.array([-14.0, -14.0, -10.0, -5.0], dtype=np.float32)
    scale = np.array([5.0, 5.0, 4.0, 3.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.uniform.cdf"] = uniform_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.uniform.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.uniform.cdf'.")


check_valid('jax.scipy.stats.uniform.cdf', generated_inputs['jax.scipy.stats.uniform.cdf'], lib="jax", suffix=0)
