
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def matrix_rank_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 full-rank square matrix (testing tol)
    M = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rtol = None
    hermitian = False
    tol = np.array(1e-5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"M": M, "rtol": rtol, "hermitian": hermitian, "tol": tol})
    )

    # Input 2: 2D float64 rank-deficient square matrix (testing tol)
    M = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    rtol = None
    hermitian = False
    tol = np.array(1e-7, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"M": M, "rtol": rtol, "hermitian": hermitian, "tol": tol})
    )

    # Input 3: 2D float32 symmetric matrix with hermitian=True (testing tol)
    M = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
    rtol = None
    hermitian = True
    tol = np.array(1e-4, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"M": M, "rtol": rtol, "hermitian": hermitian, "tol": tol})
    )

    # Input 4: 2D complex64 Hermitian matrix with hermitian=True (testing tol)
    M = np.array(
        [[2.0 + 0.0j, 1.0 - 1.0j], [1.0 + 1.0j, 3.0 + 0.0j]],
        dtype=np.complex64,
    )
    rtol = None
    hermitian = True
    tol = np.array(1e-5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"M": M, "rtol": rtol, "hermitian": hermitian, "tol": tol})
    )

    # Input 5: Rectangular tall matrix (M > N) with negative entries (testing tol)
    M = np.array(
        [[-1.0, 2.0], [3.0, -4.0], [-5.0, 6.0], [7.0, -8.0]],
        dtype=np.float32,
    )
    rtol = None
    hermitian = False
    tol = np.array(1e-6, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"M": M, "rtol": rtol, "hermitian": hermitian, "tol": tol})
    )

    # Input 6: Rectangular wide matrix (M < N) (testing tol)
    M = np.array(
        [[1.5, -2.5, 3.5, -4.5], [5.5, -6.5, 7.5, -8.5]], dtype=np.float64
    )
    rtol = None
    hermitian = False
    tol = np.array(1e-8, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"M": M, "rtol": rtol, "hermitian": hermitian, "tol": tol})
    )

    # Input 7: 3D batched square matrices (testing tol)
    M = np.random.randn(4, 3, 3).astype(np.float32)
    rtol = None
    hermitian = False
    tol = np.array(1e-5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"M": M, "rtol": rtol, "hermitian": hermitian, "tol": tol})
    )

    # Input 8: 3D batched matrices with batched tol
    M = np.random.randn(3, 4, 2).astype(np.float64)
    rtol = None
    hermitian = False
    tol = np.array([1e-5, 1e-4, 1e-3], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"M": M, "rtol": rtol, "hermitian": hermitian, "tol": tol})
    )

    # Input 9: 4D tensor representing a batch of matrices (testing tol)
    M = np.random.randn(2, 2, 4, 4).astype(np.float32)
    rtol = None
    hermitian = False
    tol = np.array(1e-6, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"M": M, "rtol": rtol, "hermitian": hermitian, "tol": tol})
    )

    # Input 10: 2D integer matrix (testing tol)
    M = np.array([[1, 0, 1], [-2, 3, 4], [0, 1, 2]], dtype=np.int32)
    rtol = None
    hermitian = False
    tol = np.array(0.1, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"M": M, "rtol": rtol, "hermitian": hermitian, "tol": tol})
    )

    # Input 11: All-zero matrix (testing tol)
    M = np.zeros((4, 4), dtype=np.float32)
    rtol = None
    hermitian = False
    tol = np.array(1e-5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"M": M, "rtol": rtol, "hermitian": hermitian, "tol": tol})
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.matrix_rank_2"] = matrix_rank_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matrix_rank_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matrix_rank_2'.")


check_valid('jax.numpy.linalg.matrix_rank', generated_inputs['jax.numpy.linalg.matrix_rank_2'], lib="jax", suffix=2)
