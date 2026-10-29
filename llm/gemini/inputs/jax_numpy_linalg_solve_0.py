
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def solve_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D matrix with 1D vector (float32)
    a = (np.eye(3) + 0.1 * np.random.randn(3, 3)).astype(np.float32)
    b = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 2: 2D matrix with 2D multiple right-hand sides (float64)
    a = (np.eye(4) * 5.0 + np.random.randn(4, 4)).astype(np.float64)
    b = np.random.randn(4, 2).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 3: Batched 3D matrix with 3D RHS column vectors (float32)
    a = (
        np.broadcast_to(np.eye(3), (2, 3, 3)) + 0.05 * np.random.randn(2, 3, 3)
    ).astype(np.float32)
    b = np.random.randn(2, 3, 1).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 4: Batched 3D matrix with 3D RHS multiple columns (float32)
    a = (
        np.broadcast_to(np.eye(2), (5, 2, 2)) + 0.05 * np.random.randn(5, 2, 2)
    ).astype(np.float32)
    b = np.random.randn(5, 2, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 5: Complex numbers (complex128)
    real_a = np.eye(3) * 3.0 + np.random.randn(3, 3)
    imag_a = np.random.randn(3, 3)
    a = (real_a + 1j * imag_a).astype(np.complex128)
    b = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 6: Batched complex numbers (complex64)
    real_a = np.broadcast_to(np.eye(4) * 4.0, (2, 4, 4)) + np.random.randn(
        2, 4, 4
    )
    imag_a = np.random.randn(2, 4, 4)
    a = (real_a + 1j * imag_a).astype(np.complex64)
    b = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 7: 4D batched matrices (float32)
    a = (
        np.broadcast_to(np.eye(5) * 2.0, (2, 3, 5, 5))
        + 0.1 * np.random.randn(2, 3, 5, 5)
    ).astype(np.float32)
    b = np.random.randn(2, 3, 5, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 8: Minimal 1x1 matrix and 1D vector (float32)
    a = np.array([[4.5]], dtype=np.float32)
    b = np.array([-9.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 9: Broadcasting batch dimensions (float64)
    a = (np.eye(3)[None, :, :] * 3.0 + 0.1 * np.random.randn(1, 3, 3)).astype(
        np.float64
    )
    b = np.random.randn(4, 3, 2).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    # Input 10: Larger 10x10 system with negative values (float64)
    a = (np.eye(10) * 10.0 - np.random.uniform(0.1, 0.5, (10, 10))).astype(
        np.float64
    )
    b = np.random.uniform(-5.0, 5.0, (10,)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "b": b}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.solve"] = solve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.solve'.")


check_valid('jax.numpy.linalg.solve', generated_inputs['jax.numpy.linalg.solve'], lib="jax", suffix=0)
