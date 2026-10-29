
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logistic_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (float32)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays (float32), matching shapes
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 1D arrays with float64
    x = np.linspace(-5.0, 5.0, 10, dtype=np.float64)
    loc = np.array([-1.0], dtype=np.float64)
    scale = np.array([2.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: 2D array for x, 0D for loc and scale
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array(1.5, dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting 2D with (1, 4) and (3, 1)
    x = np.zeros((3, 4), dtype=np.float32)
    loc = np.array([[0.0, 1.0, -1.0, 2.0]], dtype=np.float32)
    scale = np.array([[0.5], [1.0], [2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: 3D arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.abs(np.random.randn(2, 3, 4)).astype(np.float32) + 0.1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Negative loc and various scale values
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    loc = np.array([-2.5, -2.5, -2.5, -2.5, -2.5], dtype=np.float64)
    scale = np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: 4D arrays with broadcasting
    x = np.random.randn(2, 1, 4, 5).astype(np.float32)
    loc = np.zeros((1, 3, 1, 1), dtype=np.float32)
    scale = np.ones((1, 1, 4, 1), dtype=np.float32) * 1.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Large positive and negative values (float64)
    x = np.array([[-100.0, 0.0, 100.0]], dtype=np.float64)
    loc = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    scale = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: Single-element 1D arrays
    x = np.array([3.14], dtype=np.float32)
    loc = np.array([1.59], dtype=np.float32)
    scale = np.array([2.65], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.logistic.pdf_1"] = logistic_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.pdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.pdf_1'.")


check_valid('jax.scipy.stats.logistic.pdf', generated_inputs['jax.scipy.stats.logistic.pdf_1'], lib="jax", suffix=1)
