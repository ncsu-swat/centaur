
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def qr_multiply_inputs():
    list_of_inputs = []

    # Input 1: 1D c, mode='right', M > N, float32
    # a: (M, N) = (4, 2), K = min(4, 2) = 2. mode='right' requires c to have trailing dimension M=4 or 1D shape (4,)
    a = np.array(
        [[1.0, -1.0], [2.0, 3.0], [-1.0, 0.5], [0.0, 2.0]], dtype=np.float32
    )
    c = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "right",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D c, mode='left', M > N, float32
    # a: (M, N) = (4, 2), K = 2. mode='left' requires c to have leading dimension K=2 or 1D shape (2,)
    a = np.random.randn(4, 2).astype(np.float32)
    c = np.array([-1.5, 2.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "left",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D c, mode='right', float64
    # a: (5, 3), M=5. c: shape (P, M) = (2, 5)
    a = np.random.randn(5, 3).astype(np.float64)
    c = np.random.randn(2, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "right",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D c, mode='left', float64
    # a: (5, 3), K = min(5, 3) = 3. c: shape (K, P) = (3, 4)
    a = np.random.randn(5, 3).astype(np.float64)
    c = np.random.randn(3, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "left",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Square matrix M == N, mode='right', float32
    # a: (3, 3), M=3, K=3. c: (3, 3)
    a = np.random.randn(3, 3).astype(np.float32)
    c = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "right",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": True,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Fat matrix M < N, mode='left', float32
    # a: (2, 5), K = min(2, 5) = 2. c: shape (K, P) = (2, 3)
    a = np.random.randn(2, 5).astype(np.float32)
    c = np.random.randn(2, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "left",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Fat matrix M < N, mode='right', float64
    # a: (2, 5), M = 2. c: shape (P, M) = (4, 2)
    a = np.random.randn(2, 5).astype(np.float64)
    c = np.random.randn(4, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "right",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64 matrix with conjugate=True, mode='right'
    # a: (4, 3), M=4. c: shape (2, 4)
    a = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(
        np.complex64
    )
    c = (np.random.randn(2, 4) + 1j * np.random.randn(2, 4)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "c": c,
        "mode": "right",
        "pivoting": False,
        "conjugate": True,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex128 matrix with conjugate=True, mode='left'
    # a: (4, 3), K=3. c: shape (3, 2)
    a = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(
        np.complex128
    )
    c = (np.random.randn(3, 2) + 1j * np.random.randn(3, 2)).astype(
        np.complex128
    )
    input_dict = {
        "a": a,
        "c": c,
        "mode": "left",
        "pivoting": False,
        "conjugate": True,
        "overwrite_a": True,
        "overwrite_c": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batched input, mode='right', float32
    # a: (2, 4, 3), M=4, N=3. c: shape (2, 5, 4)
    a = np.random.randn(2, 4, 3).astype(np.float32)
    c = np.random.randn(2, 5, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "right",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Batched input, mode='left', float64
    # a: (3, 5, 2), K=2. c: shape (3, 2, 4)
    a = np.random.randn(3, 5, 2).astype(np.float64)
    c = np.random.randn(3, 2, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "left",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.qr_multiply"] = qr_multiply_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.qr_multiply' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.qr_multiply'.")


check_valid('jax.scipy.linalg.qr_multiply', generated_inputs['jax.scipy.linalg.qr_multiply'], lib="jax", suffix=0)
