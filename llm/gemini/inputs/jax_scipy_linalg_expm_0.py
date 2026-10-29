
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expm_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 float32 matrix
    A = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "upper_triangular": False, "max_squarings": 16}
        )
    )

    # Input 2: 3x3 float64 matrix with upper_triangular=True
    A = np.triu(np.array([[2.0, -1.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, -2.0]], dtype=np.float64))
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "upper_triangular": True, "max_squarings": 16}
        )
    )

    # Input 3: Batched 3D matrix (batch size 4, 3x3)
    A = np.random.randn(4, 3, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "upper_triangular": False, "max_squarings": 12}
        )
    )

    # Input 4: Complex64 2x2 matrix
    A = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "upper_triangular": False, "max_squarings": 8}
        )
    )

    # Input 5: Complex128 4x4 matrix, upper triangular
    A_raw = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex128)
    A = np.triu(A_raw)
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "upper_triangular": True, "max_squarings": 20}
        )
    )

    # Input 6: Zero matrix
    A = np.zeros((5, 5), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "upper_triangular": False, "max_squarings": 16}
        )
    )

    # Input 7: Identity matrix with negative max_squarings or 0 squarings
    A = np.eye(3, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "upper_triangular": False, "max_squarings": 0}
        )
    )

    # Input 8: Multi-dimensional batch (2, 3, 4, 4)
    A = np.random.randn(2, 3, 4, 4).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "upper_triangular": False, "max_squarings": 14}
        )
    )

    # Input 9: Strictly negative float32 matrix
    A = -np.abs(np.random.randn(3, 3)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "upper_triangular": False, "max_squarings": 10}
        )
    )

    # Input 10: 1x1 matrix (scalar-like)
    A = np.array([[-2.5]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "upper_triangular": True, "max_squarings": 16}
        )
    )

    # Input 11: Batched upper-triangular complex matrix
    A_batch = (np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)).astype(np.complex64)
    A = np.array([np.triu(mat) for mat in A_batch])
    list_of_inputs.append(
        copy.deepcopy(
            {"A": A, "upper_triangular": True, "max_squarings": 18}
        )
    )

    return list_of_inputs


generated_inputs["jax.scipy.linalg.expm"] = expm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.expm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.expm'.")


check_valid('jax.scipy.linalg.expm', generated_inputs['jax.scipy.linalg.expm'], lib="jax", suffix=0)
