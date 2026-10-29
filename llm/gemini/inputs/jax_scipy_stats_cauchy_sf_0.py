
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_sf_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalars
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1-D arrays with positive values
    x = np.array([0.1, 1.0, 5.0, 10.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: Negative values for x and loc
    x = np.array([-10.0, -2.5, -0.5], dtype=np.float64)
    loc = np.array([-1.0, -2.0, 0.0], dtype=np.float64)
    scale = np.array([1.5, 0.5, 2.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: 2-D arrays
    x = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)
    scale = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting - 2-D x with 1-D loc and scalar scale
    x = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    scale = np.array(1.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: 3-D arrays
    x = np.ones((2, 3, 4), dtype=np.float32) * 1.5
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Float16 dtype
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float16)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float16)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float16)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Extreme tail values
    x = np.array([1e5, -1e5, 1e10, -1e10], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Non-trivial broadcasting between different shapes
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    loc = np.array([0.0, -1.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: 4-D array
    x = np.zeros((1, 2, 1, 3), dtype=np.float32)
    loc = np.ones((1, 2, 1, 3), dtype=np.float32)
    scale = np.full((1, 2, 1, 3), 0.1, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.sf"] = cauchy_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.sf'.")


check_valid('jax.scipy.stats.cauchy.sf', generated_inputs['jax.scipy.stats.cauchy.sf'], lib="jax", suffix=0)
