
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cond_inputs():
    list_of_inputs = []

    # Input 1: 2x2 float32 matrix
    x = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "p": "fro"}))

    # Input 2: 3x3 float64 matrix with negative values
    x = np.array(
        [[-3.0, 1.0, 0.5], [1.0, -4.0, 2.0], [0.5, 2.0, -5.0]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "p": "fro"}))

    # Input 3: 4x4 float32 diagonally dominant matrix
    x = (np.eye(4) * 5.0 + np.random.randn(4, 4) * 0.1).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "p": "fro"}))

    # Input 4: 3D batched float32 matrices (batch size 2, 3x3)
    rng = np.random.default_rng(42)
    x = (
        np.stack([np.eye(3), np.eye(3)]) * 3.0
        + rng.standard_normal((2, 3, 3)) * 0.2
    ).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "p": "fro"}))

    # Input 5: 3D batched float64 matrices (batch size 4, 2x2)
    x = (
        np.stack([np.eye(2)] * 4) * 2.0 + rng.standard_normal((4, 2, 2)) * 0.1
    ).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "p": "fro"}))

    # Input 6: 4D batched float32 matrices (shape (2, 3, 2, 2))
    x = (
        np.broadcast_to(np.eye(2), (2, 3, 2, 2)) * 4.0
        + rng.standard_normal((2, 3, 2, 2)) * 0.2
    ).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "p": "fro"}))

    # Input 7: 2x2 complex64 matrix
    x = np.array(
        [[2.0 + 1.0j, 0.5 - 0.5j], [0.5 + 0.5j, 3.0 + 0.0j]],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "p": "fro"}))

    # Input 8: 3x3 complex128 matrix
    x = (
        np.eye(3, dtype=np.complex128) * (3.0 + 2.0j)
        + (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))) * 0.1
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "p": "fro"}))

    # Input 9: 5x5 float32 well-conditioned symmetric matrix
    a = rng.standard_normal((5, 5)).astype(np.float32)
    x = a.T @ a + np.eye(5, dtype=np.float32) * 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "p": "fro"}))

    # Input 10: 3D batched complex64 matrices (batch size 3, 3x3)
    real_part = (
        np.broadcast_to(np.eye(3), (3, 3, 3)) * 3.0
        + rng.standard_normal((3, 3, 3)) * 0.1
    )
    imag_part = rng.standard_normal((3, 3, 3)) * 0.1
    x = (real_part + 1j * imag_part).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "p": "fro"}))

    # Input 11: 6x6 float64 invertible matrix
    x = np.diag(np.linspace(1.0, 10.0, 6)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "p": "fro"}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.cond_3"] = cond_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.cond_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.cond_3'.")


check_valid('jax.numpy.linalg.cond', generated_inputs['jax.numpy.linalg.cond_3'], lib="jax", suffix=3)
