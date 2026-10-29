
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def eigvalsh_inputs():
    list_of_inputs = []

    # Input 1: 2D real float32 symmetric matrix, UPLO='L', symmetrize_input=True
    mat = np.random.randn(3, 3).astype(np.float32)
    a = (mat + mat.T) / 2.0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "UPLO": "L", "symmetrize_input": True})
    )

    # Input 2: 2D real float64 symmetric matrix, UPLO='U', symmetrize_input=False
    mat = np.random.randn(4, 4).astype(np.float64)
    a = (mat + mat.T) / 2.0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "UPLO": "U", "symmetrize_input": False})
    )

    # Input 3: 2D complex64 Hermitian matrix, UPLO='L', symmetrize_input=True
    mat = (
        np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    ).astype(np.complex64)
    a = (mat + mat.conj().T) / 2.0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "UPLO": "L", "symmetrize_input": True})
    )

    # Input 4: 2D complex128 Hermitian matrix, UPLO='U', symmetrize_input=False
    mat = (
        np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    ).astype(np.complex128)
    a = (mat + mat.conj().T) / 2.0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "UPLO": "U", "symmetrize_input": False})
    )

    # Input 5: 3D batched real symmetric matrix (batch of 2), UPLO='L', symmetrize_input=True
    mat = np.random.randn(2, 3, 3).astype(np.float32)
    a = (mat + np.swapaxes(mat, -1, -2)) / 2.0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "UPLO": "L", "symmetrize_input": True})
    )

    # Input 6: 4D batched real symmetric matrix, UPLO='U', symmetrize_input=True
    mat = np.random.randn(2, 2, 5, 5).astype(np.float32)
    a = (mat + np.swapaxes(mat, -1, -2)) / 2.0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "UPLO": "U", "symmetrize_input": True})
    )

    # Input 7: 1x1 matrix edge case, float32, UPLO='L', symmetrize_input=False
    a = np.array([[-5.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "UPLO": "L", "symmetrize_input": False})
    )

    # Input 8: 3D batched complex Hermitian matrix, UPLO='U', symmetrize_input=True
    mat = (
        np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)
    ).astype(np.complex64)
    a = (mat + np.swapaxes(mat, -1, -2).conj()) / 2.0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "UPLO": "U", "symmetrize_input": True})
    )

    # Input 9: Real matrix with strictly negative eigenvalues, float64, UPLO='L', symmetrize_input=True
    mat = np.random.randn(4, 4).astype(np.float64)
    a = -(mat @ mat.T) - np.eye(4, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "UPLO": "L", "symmetrize_input": True})
    )

    # Input 10: Larger 10x10 float32 symmetric matrix, UPLO='U', symmetrize_input=False
    mat = np.random.randn(10, 10).astype(np.float32)
    a = (mat + mat.T) / 2.0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "UPLO": "U", "symmetrize_input": False})
    )

    # Input 11: Real matrix with zeros and identity structure, float64, UPLO='L', symmetrize_input=False
    a = np.eye(6, dtype=np.float64) * 2.0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "UPLO": "L", "symmetrize_input": False})
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.eigvalsh"] = eigvalsh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.eigvalsh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.eigvalsh'.")


check_valid('jax.numpy.linalg.eigvalsh', generated_inputs['jax.numpy.linalg.eigvalsh'], lib="jax", suffix=0)
