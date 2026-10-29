
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def solve_sylvester_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 real float32 matrices with "schur" method
    A = np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32)
    B = np.array([[4.0, 1.0], [2.0, 5.0]], dtype=np.float32)
    C = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        {
            "A": A,
            "B": B,
            "C": C,
            "method": "schur",
            "tol": 1e-8,
        }
    )

    # Input 2: Different m and n dimensions, float64 matrices with "schur"
    A = (
        np.array(
            [[3.0, -1.0, 0.0], [0.5, 4.0, 1.0], [-1.0, 0.0, 5.0]],
            dtype=np.float64,
        )
        + np.eye(3) * 2.0
    )
    B = (
        np.array(
            [[2.0, 1.0, -0.5, 0.0], [0.0, 3.0, 1.0, -1.0], [1.0, 0.0, 4.0, 0.5], [-0.5, 1.0, 0.0, 5.0]],
            dtype=np.float64,
        )
        + np.eye(4) * 2.0
    )
    C = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(
        {
            "A": A,
            "B": B,
            "C": C,
            "method": "schur",
            "tol": 1e-7,
        }
    )

    # Input 3: Symmetric positive definite matrices with "eigen" method
    M1 = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float32)
    M2 = np.array([[3.0, -0.2], [-0.2, 2.5]], dtype=np.float32)
    A = M1 @ M1.T + np.eye(2, dtype=np.float32)
    B = M2 @ M2.T + np.eye(2, dtype=np.float32)
    C = np.array([[5.0, -1.0], [2.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(
        {
            "A": A,
            "B": B,
            "C": C,
            "method": "eigen",
            "tol": 1e-8,
        }
    )

    # Input 4: Distinct m > n dimensions with "eigen" method, float64
    rng = np.random.RandomState(42)
    M_A = rng.randn(4, 4)
    A = (M_A @ M_A.T + 3.0 * np.eye(4)).astype(np.float64)
    M_B = rng.randn(2, 2)
    B = (M_B @ M_B.T + 3.0 * np.eye(2)).astype(np.float64)
    C = rng.randn(4, 2).astype(np.float64)
    list_of_inputs.append(
        {
            "A": A,
            "B": B,
            "C": C,
            "method": "eigen",
            "tol": 1e-6,
        }
    )

    # Input 5: Batched 3D inputs (batch_size=2, m=3, n=3) with "schur" method
    A = np.zeros((2, 3, 3), dtype=np.float32)
    B = np.zeros((2, 3, 3), dtype=np.float32)
    C = np.zeros((2, 3, 3), dtype=np.float32)
    for i in range(2):
        A[i] = np.eye(3, dtype=np.float32) * (3.0 + i) + rng.randn(3, 3) * 0.1
        B[i] = np.eye(3, dtype=np.float32) * (4.0 + i) + rng.randn(3, 3) * 0.1
        C[i] = rng.randn(3, 3).astype(np.float32)
    list_of_inputs.append(
        {
            "A": A,
            "B": B,
            "C": C,
            "method": "schur",
            "tol": 1e-8,
        }
    )

    # Input 6: Batched 3D inputs with non-square C (batch_size=3, m=2, n=4) with "eigen"
    A = np.tile(np.eye(2, dtype=np.float64) * 4.0, (3, 1, 1)) + rng.randn(3, 2, 2) * 0.1
    B = np.tile(np.eye(4, dtype=np.float64) * 5.0, (3, 1, 1)) + rng.randn(3, 4, 4) * 0.1
    C = rng.randn(3, 2, 4).astype(np.float64)
    list_of_inputs.append(
        {
            "A": A,
            "B": B,
            "C": C,
            "method": "eigen",
            "tol": 1e-8,
        }
    )

    # Input 7: Complex64 matrices with "schur" method
    A = (np.array([[2.0 + 1.0j, 0.5 - 0.2j], [-0.1 + 0.3j, 3.0 + 0.0j]], dtype=np.complex64) + np.eye(2) * 3.0)
    B = (np.array([[4.0 - 0.5j, 0.2 + 0.1j], [0.3 - 0.2j, 2.5 + 1.0j]], dtype=np.complex64) + np.eye(2) * 3.0)
    C = np.array([[1.0 + 2.0j, -0.5 + 0.5j], [0.0 - 1.0j, 2.0 + 0.0j]], dtype=np.complex64)
    list_of_inputs.append(
        {
            "A": A,
            "B": B,
            "C": C,
            "method": "schur",
            "tol": 1e-8,
        }
    )

    # Input 8: Complex128 matrices with "eigen" method
    A = (rng.randn(3, 3) + 1j * rng.randn(3, 3) + np.eye(3) * 6.0).astype(np.complex128)
    B = (rng.randn(2, 2) + 1j * rng.randn(2, 2) + np.eye(2) * 6.0).astype(np.complex128)
    C = (rng.randn(3, 2) + 1j * rng.randn(3, 2)).astype(np.complex128)
    list_of_inputs.append(
        {
            "A": A,
            "B": B,
            "C": C,
            "method": "eigen",
            "tol": 1e-6,
        }
    )

    # Input 9: Broadcastable batch dimensions (2, 1, 2, 2) and (1, 3, 2, 2) -> (2, 3, 2, 2)
    A = np.tile(np.eye(2, dtype=np.float32) * 5.0, (2, 1, 1, 1))
    B = np.tile(np.eye(2, dtype=np.float32) * 6.0, (1, 3, 1, 1))
    C = rng.randn(2, 3, 2, 2).astype(np.float32)
    list_of_inputs.append(
        {
            "A": A,
            "B": B,
            "C": C,
            "method": "schur",
            "tol": 1e-5,
        }
    )

    # Input 10: 1x1 scalar-like matrices
    A = np.array([[3.5]], dtype=np.float64)
    B = np.array([[2.5]], dtype=np.float64)
    C = np.array([[-7.0]], dtype=np.float64)
    list_of_inputs.append(
        {
            "A": A,
            "B": B,
            "C": C,
            "method": "eigen",
            "tol": 1e-9,
        }
    )

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.scipy.linalg.solve_sylvester"] = solve_sylvester_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.solve_sylvester' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.solve_sylvester'.")


check_valid('jax.scipy.linalg.solve_sylvester', generated_inputs['jax.scipy.linalg.solve_sylvester'], lib="jax", suffix=0)
