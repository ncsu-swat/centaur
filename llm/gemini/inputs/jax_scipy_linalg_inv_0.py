
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def inv_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 float32 matrix
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": True}
    )

    # Input 2: 3x3 float64 matrix with negative values
    a = np.array(
        [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]],
        dtype=np.float64,
    )
    list_of_inputs.append(
        {"a": a, "overwrite_a": True, "check_finite": False}
    )

    # Input 3: 1x1 matrix
    a = np.array([[5.5]], dtype=np.float32)
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": False}
    )

    # Input 4: 4x4 float32 diagonally dominant matrix
    rng = np.random.RandomState(42)
    m = rng.randn(4, 4).astype(np.float32)
    a = m + 5.0 * np.eye(4, dtype=np.float32)
    list_of_inputs.append(
        {"a": a, "overwrite_a": True, "check_finite": True}
    )

    # Input 5: Batched 3D matrix (batch size 2, 3x3)
    b1 = np.eye(3, dtype=np.float32) + 0.1 * rng.randn(3, 3).astype(np.float32)
    b2 = 2.0 * np.eye(3, dtype=np.float32) + 0.1 * rng.randn(3, 3).astype(
        np.float32
    )
    a = np.stack([b1, b2], axis=0)
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": True}
    )

    # Input 6: Higher dimension batch (2, 3, 2, 2)
    eye_batch = np.broadcast_to(np.eye(2, dtype=np.float64), (2, 3, 2, 2)).copy()
    noise = 0.05 * rng.randn(2, 3, 2, 2).astype(np.float64)
    a = eye_batch + noise
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": False}
    )

    # Input 7: Complex64 2x2 matrix
    real_part = rng.randn(2, 2).astype(np.float32) + 3.0 * np.eye(
        2, dtype=np.float32
    )
    imag_part = rng.randn(2, 2).astype(np.float32)
    a = (real_part + 1j * imag_part).astype(np.complex64)
    list_of_inputs.append(
        {"a": a, "overwrite_a": True, "check_finite": True}
    )

    # Input 8: Complex128 3x3 matrix
    real_part = rng.randn(3, 3).astype(np.float64) + 4.0 * np.eye(
        3, dtype=np.float64
    )
    imag_part = rng.randn(3, 3).astype(np.float64)
    a = (real_part + 1j * imag_part).astype(np.complex128)
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": True}
    )

    # Input 9: Symmetric positive definite 5x5 float64 matrix
    q = rng.randn(5, 5).astype(np.float64)
    a = q.T @ q + np.eye(5, dtype=np.float64)
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": False}
    )

    # Input 10: Batched complex matrix (4, 2, 2)
    real_batch = (
        np.broadcast_to(np.eye(2, dtype=np.float32), (4, 2, 2)).copy()
        + 0.1 * rng.randn(4, 2, 2).astype(np.float32)
    )
    imag_batch = 0.1 * rng.randn(4, 2, 2).astype(np.float32)
    a = (real_batch + 1j * imag_batch).astype(np.complex64)
    list_of_inputs.append(
        {"a": a, "overwrite_a": True, "check_finite": False}
    )

    # Input 11: 6x6 float32 upper triangular matrix with non-zero diagonal
    tri = np.triu(rng.randn(6, 6).astype(np.float32))
    np.fill_diagonal(tri, np.abs(np.diag(tri)) + 1.0)
    a = tri
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": True}
    )

    return list_of_inputs


generated_inputs["jax.scipy.linalg.inv"] = inv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.inv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.inv'.")


check_valid('jax.scipy.linalg.inv', generated_inputs['jax.scipy.linalg.inv'], lib="jax", suffix=0)
