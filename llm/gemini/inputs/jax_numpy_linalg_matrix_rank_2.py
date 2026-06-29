
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_linalg_matrix_rank_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 square matrix, rtol as scalar tensor
    M1 = np.random.randn(5, 5).astype(np.float32)
    rtol1 = np.array(1e-4, dtype=np.float32)
    list_of_inputs.append({
        "M": M1,
        "rtol": rtol1,
        "hermitian": False,
        "tol": None
    })

    # Input 2: float64 rectangular matrix, tol as scalar tensor
    M2 = np.random.randn(6, 4).astype(np.float64)
    tol2 = np.array(1e-6, dtype=np.float64)
    list_of_inputs.append({
        "M": M2,
        "rtol": None,
        "hermitian": False,
        "tol": tol2
    })

    # Input 3: Hermitian complex matrix, rtol as scalar tensor
    B = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    M3 = B + B.conj().T
    rtol3 = np.array(1e-5, dtype=np.float32)
    list_of_inputs.append({
        "M": M3,
        "rtol": rtol3,
        "hermitian": True,
        "tol": None
    })

    # Input 4: Rank deficient matrix (rank 1), tol as scalar tensor
    u = np.random.randn(5, 1).astype(np.float32)
    v = np.random.randn(1, 5).astype(np.float32)
    M4 = np.dot(u, v)
    tol4 = np.array(1e-3, dtype=np.float32)
    list_of_inputs.append({
        "M": M4,
        "rtol": None,
        "hermitian": False,
        "tol": tol4
    })

    # Input 5: Batch of matrices (shape 2, 3, 3), rtol as scalar tensor
    M5 = np.random.randn(2, 3, 3).astype(np.float32)
    rtol5 = np.array(1e-4, dtype=np.float32)
    list_of_inputs.append({
        "M": M5,
        "rtol": rtol5,
        "hermitian": False,
        "tol": None
    })

    # Input 6: Batch of matrices with a batch of tolerances
    M6 = np.random.randn(3, 4, 5).astype(np.float32)
    rtol6 = np.array([1e-4, 1e-5, 1e-6], dtype=np.float32)
    list_of_inputs.append({
        "M": M6,
        "rtol": rtol6,
        "hermitian": False,
        "tol": None
    })

    # Input 7: Hermitian real matrix, tol as scalar tensor
    B = np.random.randn(5, 5).astype(np.float32)
    M7 = B + B.T
    tol7 = np.array(1e-5, dtype=np.float32)
    list_of_inputs.append({
        "M": M7,
        "rtol": None,
        "hermitian": True,
        "tol": tol7
    })

    # Input 8: Integer matrix, rtol and tol as None
    M8 = np.random.randint(-10, 10, size=(4, 4)).astype(np.int32)
    list_of_inputs.append({
        "M": M8,
        "rtol": None,
        "hermitian": False,
        "tol": None
    })

    # Input 9: Large rectangular matrix (100, 50), rtol as scalar tensor
    M9 = np.random.randn(100, 50).astype(np.float32)
    rtol9 = np.array(1e-5, dtype=np.float32)
    list_of_inputs.append({
        "M": M9,
        "rtol": rtol9,
        "hermitian": False,
        "tol": None
    })

    # Input 10: 4D batch of matrices, tol as scalar tensor
    M10 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    tol10 = np.array(1e-4, dtype=np.float32)
    list_of_inputs.append({
        "M": M10,
        "rtol": None,
        "hermitian": False,
        "tol": tol10
    })

    return list_of_inputs

generated_inputs["jax.numpy.linalg.matrix_rank_2"] = jax_numpy_linalg_matrix_rank_inputs()

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
