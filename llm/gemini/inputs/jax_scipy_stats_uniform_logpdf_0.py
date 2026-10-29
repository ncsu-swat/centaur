
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def uniform_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays, all within [loc, loc + scale]
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 0D scalar arrays
    x = np.array(0.5, dtype=np.float64)
    loc = np.array(0.0, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 2D float32 arrays with negative loc
    x = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    loc = np.array([[-2.0, -1.0], [0.0, 1.0]], dtype=np.float32)
    scale = np.array([[3.0, 3.0], [2.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting loc and scale against x
    x = np.linspace(0, 10, 20, dtype=np.float32).reshape(4, 5)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([10.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Values outside [loc, loc + scale] (resulting in -inf)
    x = np.array([-5.0, 0.0, 5.0, 15.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: 3D float32 arrays
    x = np.random.uniform(2.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    loc = np.full((2, 3, 4), 2.0, dtype=np.float32)
    scale = np.full((2, 3, 4), 3.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Integer arrays
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    loc = np.array([0, 1, 2, 3], dtype=np.int32)
    scale = np.array([5, 5, 5, 5], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Multi-dimensional broadcasting with differing shapes
    x = np.ones((3, 1, 4), dtype=np.float64) * 0.5
    loc = np.zeros((1, 2, 1), dtype=np.float64)
    scale = np.ones((3, 2, 4), dtype=np.float64) * 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Large and small scale values
    x = np.array([1e-5, 1e5], dtype=np.float64)
    loc = np.array([0.0, 0.0], dtype=np.float64)
    scale = np.array([1e-4, 1e6], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: 1D array with negative range
    x = np.array([-10.0, -8.0, -6.0, -4.0], dtype=np.float32)
    loc = np.array([-10.0, -10.0, -10.0, -10.0], dtype=np.float32)
    scale = np.array([8.0, 8.0, 8.0, 8.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.uniform.logpdf"] = uniform_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.uniform.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.uniform.logpdf'.")


check_valid('jax.scipy.stats.uniform.logpdf', generated_inputs['jax.scipy.stats.uniform.logpdf'], lib="jax", suffix=0)
