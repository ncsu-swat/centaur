
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_ppf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, standard Cauchy
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 2: 0D scalar arrays, float32
    q = np.array(0.5, dtype=np.float32)
    loc = np.array(2.0, dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays, float64, negative loc
    q = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64)
    loc = np.array([[-1.5, -0.5], [0.5, 1.5]], dtype=np.float64)
    scale = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 4: 3D arrays, float32
    q = np.full((2, 3, 4), 0.3, dtype=np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting shapes (3, 1) and (1, 4)
    q = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)
    loc = np.array([[0.0, -1.0, 1.0, 2.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0, 0.5, 1.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 6: 1D array with values close to 0 and 1, float64
    q = np.array([0.001, 0.01, 0.05, 0.95, 0.99, 0.999], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 7: Broadcasting 0D loc and scale with 2D q
    q = np.array([[0.1, 0.3, 0.5], [0.7, 0.8, 0.9]], dtype=np.float32)
    loc = np.array(-5.0, dtype=np.float32)
    scale = np.array(10.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 8: 1D single-element arrays
    q = np.array([0.75], dtype=np.float32)
    loc = np.array([3.0], dtype=np.float32)
    scale = np.array([2.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 9: 4D arrays, float64
    q = np.linspace(0.1, 0.9, 12).reshape(1, 2, 2, 3).astype(np.float64)
    loc = np.zeros((1, 2, 2, 3), dtype=np.float64)
    scale = np.ones((1, 2, 2, 3), dtype=np.float64) * 0.5
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 10: 2D arrays with fractional scale and large negative loc
    q = np.array([[0.25, 0.5], [0.5, 0.75]], dtype=np.float32)
    loc = np.array([[-100.0, -50.0], [-10.0, 0.0]], dtype=np.float32)
    scale = np.array([[0.01, 0.05], [0.1, 0.2]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.ppf_1"] = cauchy_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.ppf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.ppf_1'.")


check_valid('jax.scipy.stats.cauchy.ppf', generated_inputs['jax.scipy.stats.cauchy.ppf_1'], lib="jax", suffix=1)
