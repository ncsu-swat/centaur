
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class SafeTuple(tuple):
    def __array__(self, dtype=None, copy=None):
        c, lower = self
        return np.array([np.min(c), float(lower)], dtype=np.float64)

def cho_solve_inputs():
    list_of_inputs = []

    def make_chol(N, lower=True, dtype=np.float32):
        A = np.random.randn(N, N).astype(dtype)
        A = np.matmul(A, A.T) + N * np.eye(N, dtype=dtype)
        c = np.linalg.cholesky(A)
        if not lower:
            c = c.T.copy()
        return c

    def make_chol_batch(batch_shape, N, lower=True, dtype=np.float32):
        A = np.random.randn(*(batch_shape + (N, N))).astype(dtype)
        A_T = np.swapaxes(A, -1, -2)
        A = np.matmul(A, A_T) + N * np.eye(N, dtype=dtype)
        c = np.linalg.cholesky(A)
        if not lower:
            c = np.swapaxes(c, -1, -2).copy()
        return c

    # Input 1: Basic real float32, lower=True, 1D RHS
    c1 = make_chol(3, lower=True, dtype=np.float32)
    b1 = np.random.randn(3).astype(np.float32)
    list_of_inputs.append({
        "c_and_lower": SafeTuple((c1, True)),
        "b": b1,
        "overwrite_b": False,
        "check_finite": True
    })

    # Input 2: Basic real float32, lower=False, 2D RHS
    c2 = make_chol(4, lower=False, dtype=np.float32)
    b2 = np.random.randn(4, 2).astype(np.float32)
    list_of_inputs.append({
        "c_and_lower": SafeTuple((c2, False)),
        "b": b2,
        "overwrite_b": True,
        "check_finite": False
    })

    # Input 3: Float64 precision, lower=True, 1D RHS
    c3 = make_chol(2, lower=True, dtype=np.float64)
    b3 = np.random.randn(2).astype(np.float64)
    list_of_inputs.append({
        "c_and_lower": SafeTuple((c3, True)),
        "b": b3,
        "overwrite_b": False,
        "check_finite": False
    })

    # Input 4: Float64 precision, lower=False, 2D RHS
    c4 = make_chol(5, lower=False, dtype=np.float64)
    b4 = np.random.randn(5, 3).astype(np.float64)
    list_of_inputs.append({
        "c_and_lower": SafeTuple((c4, False)),
        "b": b4,
        "overwrite_b": True,
        "check_finite": True
    })

    # Input 5: Batched 1D, lower=True
    c5 = make_chol_batch((2,), 3, lower=True, dtype=np.float32)
    b5 = np.random.randn(2, 3, 1).astype(np.float32)
    list_of_inputs.append({
        "c_and_lower": SafeTuple((c5, True)),
        "b": b5,
        "overwrite_b": False,
        "check_finite": True
    })

    # Input 6: Batched 2D, lower=False
    c6 = make_chol_batch((4,), 2, lower=False, dtype=np.float32)
    b6 = np.random.randn(4, 2, 3).astype(np.float32)
    list_of_inputs.append({
        "c_and_lower": SafeTuple((c6, False)),
        "b": b6,
        "overwrite_b": True,
        "check_finite": True
    })

    # Input 7: Larger matrix size, lower=True
    c7 = make_chol(10, lower=True, dtype=np.float32)
    b7 = np.random.randn(10, 5).astype(np.float32)
    list_of_inputs.append({
        "c_and_lower": SafeTuple((c7, True)),
        "b": b7,
        "overwrite_b": False,
        "check_finite": False
    })

    # Input 8: Smallest 1x1 matrix size, lower=False
    c8 = make_chol(1, lower=False, dtype=np.float32)
    b8 = np.random.randn(1).astype(np.float32)
    list_of_inputs.append({
        "c_and_lower": SafeTuple((c8, False)),
        "b": b8,
        "overwrite_b": False,
        "check_finite": True
    })

    # Input 9: Multi-dimensional batching
    c9 = make_chol_batch((2, 3), 4, lower=True, dtype=np.float32)
    b9 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append({
        "c_and_lower": SafeTuple((c9, True)),
        "b": b9,
        "overwrite_b": False,
        "check_finite": True
    })

    # Input 10: High-dimensional float64
    c10 = make_chol(15, lower=False, dtype=np.float64)
    b10 = np.random.randn(15).astype(np.float64)
    list_of_inputs.append({
        "c_and_lower": SafeTuple((c10, False)),
        "b": b10,
        "overwrite_b": True,
        "check_finite": False
    })

    return list_of_inputs

generated_inputs["jax.scipy.linalg.cho_solve"] = cho_solve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.cho_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.cho_solve'.")


check_valid('jax.scipy.linalg.cho_solve', generated_inputs['jax.scipy.linalg.cho_solve'], lib="jax", suffix=0)
