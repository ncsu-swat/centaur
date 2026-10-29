
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expm_frechet_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 float32 matrices
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    E = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "E": E, "method": "blockEnlarge", "compute_expm": True}
        )
    )

    # Input 2: 3x3 float64 matrices with compute_expm=False
    A = np.random.randn(3, 3).astype(np.float64)
    E = np.random.randn(3, 3).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"A": A, "E": E, "method": "SPS", "compute_expm": False})
    )

    # Input 3: 1x1 matrix (scalar matrix)
    A = np.array([[-2.5]], dtype=np.float32)
    E = np.array([[0.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "E": E, "method": "default", "compute_expm": True}
        )
    )

    # Input 4: Batched 3D tensor (batch size 4, 2x2 matrices)
    A = np.random.randn(4, 2, 2).astype(np.float32)
    E = np.random.randn(4, 2, 2).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"A": A, "E": E, "method": "", "compute_expm": True})
    )

    # Input 5: Complex128 2x2 matrices
    A = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(
        np.complex128
    )
    E = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(
        np.complex128
    )
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "E": E, "method": "blockEnlarge", "compute_expm": False}
        )
    )

    # Input 6: Multi-batch 4D tensor (2, 3, 3, 3)
    A = np.random.randn(2, 3, 3, 3).astype(np.float64)
    E = np.random.randn(2, 3, 3, 3).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"A": A, "E": E, "method": "pade", "compute_expm": True})
    )

    # Input 7: Zero matrix for A and non-zero E
    A = np.zeros((4, 4), dtype=np.float32)
    E = np.eye(4, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"A": A, "E": E, "method": "SPS", "compute_expm": True})
    )

    # Input 8: Complex64 3x3 matrices
    A = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex64
    )
    E = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex64
    )
    list_of_inputs.append(
        copy.deepcopy({"A": A, "E": E, "method": "taylor", "compute_expm": True})
    )

    # Input 9: Diagonal/identity scaling with negative values
    A = -np.eye(5, dtype=np.float32) * 3.0
    E = np.ones((5, 5), dtype=np.float32) * -0.5
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "E": E, "method": "blockEnlarge", "compute_expm": False}
        )
    )

    # Input 10: Larger 8x8 float64 matrix
    A = np.random.randn(8, 8).astype(np.float64)
    E = np.random.randn(8, 8).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"A": A, "E": E, "method": "none", "compute_expm": True})
    )

    # Input 11: Batched complex128 tensor
    A = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(
        np.complex128
    )
    E = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(
        np.complex128
    )
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "E": E, "method": "blockEnlarge", "compute_expm": False}
        )
    )

    return list_of_inputs


generated_inputs["jax.scipy.linalg.expm_frechet"] = expm_frechet_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.expm_frechet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.expm_frechet'.")


check_valid('jax.scipy.linalg.expm_frechet', generated_inputs['jax.scipy.linalg.expm_frechet'], lib="jax", suffix=0)
