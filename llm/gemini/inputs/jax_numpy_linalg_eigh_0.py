
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def eigh_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 real symmetric matrix (float32)
    a1 = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        {"a": a1, "UPLO": "L", "symmetrize_input": True}
    )

    # Input 2: 3x3 real symmetric matrix (float64)
    a2 = np.array(
        [[4.0, 1.0, -2.0], [1.0, 2.0, 0.5], [-2.0, 0.5, 3.0]], dtype=np.float64
    )
    list_of_inputs.append(
        {"a": a2, "UPLO": "U", "symmetrize_input": False}
    )

    # Input 3: 2x2 complex Hermitian matrix (complex64)
    a3 = np.array(
        [[1.0 + 0.0j, -2.0j], [2.0j, 1.0 + 0.0j]], dtype=np.complex64
    )
    list_of_inputs.append(
        {"a": a3, "UPLO": "L", "symmetrize_input": True}
    )

    # Input 4: 3x3 complex Hermitian matrix (complex128)
    a4 = np.array(
        [
            [3.0 + 0.0j, 1.0 - 1.0j, 0.5 + 2.0j],
            [1.0 + 1.0j, 2.0 + 0.0j, -1.0 + 0.5j],
            [0.5 - 2.0j, -1.0 - 0.5j, 4.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    list_of_inputs.append(
        {"a": a4, "UPLO": "U", "symmetrize_input": False}
    )

    # Input 5: Batched 3D real symmetric matrices (float32)
    m = np.random.randn(5, 4, 4).astype(np.float32)
    a5 = (m + np.swapaxes(m, -1, -2)) / 2.0
    list_of_inputs.append(
        {"a": a5, "UPLO": "L", "symmetrize_input": True}
    )

    # Input 6: Batched 3D real symmetric matrices (float64)
    m = np.random.randn(3, 5, 5).astype(np.float64)
    a6 = (m + np.swapaxes(m, -1, -2)) / 2.0
    list_of_inputs.append(
        {"a": a6, "UPLO": "U", "symmetrize_input": True}
    )

    # Input 7: 4D batched complex Hermitian matrices (complex64)
    m = (
        np.random.randn(2, 3, 4, 4) + 1j * np.random.randn(2, 3, 4, 4)
    ).astype(np.complex64)
    a7 = (m + np.swapaxes(m, -1, -2).conj()) / 2.0
    list_of_inputs.append(
        {"a": a7, "UPLO": "L", "symmetrize_input": False}
    )

    # Input 8: 1x1 scalar matrix edge-case (float32)
    a8 = np.array([[-5.5]], dtype=np.float32)
    list_of_inputs.append(
        {"a": a8, "UPLO": "U", "symmetrize_input": True}
    )

    # Input 9: Diagonal-dominant matrix with negative elements (float64)
    a9 = np.array(
        [
            [-10.0, 1.0, 2.0, -1.0],
            [1.0, -8.0, 0.0, 1.0],
            [2.0, 0.0, -6.0, 3.0],
            [-1.0, 1.0, 3.0, -5.0],
        ],
        dtype=np.float64,
    )
    list_of_inputs.append(
        {"a": a9, "UPLO": "L", "symmetrize_input": False}
    )

    # Input 10: 3D batched complex Hermitian matrices (complex128)
    m = (
        np.random.randn(4, 3, 3) + 1j * np.random.randn(4, 3, 3)
    ).astype(np.complex128)
    a10 = (m + np.swapaxes(m, -1, -2).conj()) / 2.0
    list_of_inputs.append(
        {"a": a10, "UPLO": "U", "symmetrize_input": True}
    )

    # Input 11: All-zero matrix (float32)
    a11 = np.zeros((6, 6), dtype=np.float32)
    list_of_inputs.append(
        {"a": a11, "UPLO": "L", "symmetrize_input": True}
    )

    # Input 12: Identity matrix (float64)
    a12 = np.eye(8, dtype=np.float64)
    list_of_inputs.append(
        {"a": a12, "UPLO": "U", "symmetrize_input": False}
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.eigh"] = eigh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.eigh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.eigh'.")


check_valid('jax.numpy.linalg.eigh', generated_inputs['jax.numpy.linalg.eigh'], lib="jax", suffix=0)
