
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def laplace_pdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, standard Laplace
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 2D arrays, matching shapes
    x = np.array([[-1.5, 0.5], [2.0, -0.5]], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.5], [2.0, 0.8]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: Scalar tensors (0D arrays)
    x = np.array(1.5, dtype=np.float64)
    loc = np.array(-0.5, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: 3D array for x, broadcast with 1D loc and scale
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.zeros((4,), dtype=np.float32)
    scale = np.ones((4,), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcast across different dimensions (column and row vectors)
    x = np.linspace(-5.0, 5.0, 10, dtype=np.float32).reshape(10, 1)
    loc = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: float64 precision with negative location
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    loc = np.array([-5.0, -5.0, -5.0, -5.0, -5.0], dtype=np.float64)
    scale = np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: 4D tensors
    x = np.random.uniform(-3.0, 3.0, size=(2, 2, 3, 3)).astype(np.float32)
    loc = np.zeros((2, 2, 3, 3), dtype=np.float32)
    scale = np.full((2, 2, 3, 3), 1.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Large scale parameter
    x = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    loc = np.array([150.0, 150.0, 150.0], dtype=np.float32)
    scale = np.array([50.0, 100.0, 200.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Small scale parameter (narrow distribution)
    x = np.array([0.001, 0.002, 0.005], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: Broadcasting with leading singleton dimensions
    x = np.random.randn(5, 1, 4).astype(np.float32)
    loc = np.random.randn(1, 3, 4).astype(np.float32)
    scale = np.exp(np.random.randn(1, 3, 1)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.laplace.pdf"] = laplace_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.laplace.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.laplace.pdf'.")


check_valid('jax.scipy.stats.laplace.pdf', generated_inputs['jax.scipy.stats.laplace.pdf'], lib="jax", suffix=0)
