
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expon_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (float32)
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays (float32)
    x = np.array([0.5, 1.0, 2.5, 5.0], dtype=np.float32)
    loc = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays with matching shapes (float64)
    x = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    loc = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    scale = np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting 1D to 2D
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([[1.0], [2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: 3D arrays
    x = np.random.uniform(0.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: x < loc (testing CDF output of 0)
    x = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
    loc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Negative loc values
    x = np.array([-0.5, 0.0, 1.5, 3.0], dtype=np.float64)
    loc = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float64)
    scale = np.array([0.1, 0.5, 1.0, 2.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: 4D tensors
    x = np.random.uniform(1.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.zeros((1, 2, 1, 2), dtype=np.float32)
    scale = np.full((2, 1, 2, 1), 2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Large scale and large x
    x = np.array([100.0, 500.0, 1000.0], dtype=np.float32)
    loc = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    scale = np.array([50.0, 200.0, 500.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: 1D array x with scalar (0D) loc and scale
    x = np.linspace(0.0, 10.0, 15, dtype=np.float64)
    loc = np.array(0.0, dtype=np.float64)
    scale = np.array(3.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.expon.cdf"] = expon_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.cdf'.")


check_valid('jax.scipy.stats.expon.cdf', generated_inputs['jax.scipy.stats.expon.cdf'], lib="jax", suffix=0)
