
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def matrix_rank_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 matrix with rtol
    M1 = np.random.randn(5, 5).astype(np.float32)
    list_of_inputs.append({
        "M": M1,
        "rtol": 1e-5,
        "hermitian": False,
        "tol": None
    })

    # Input 2: Symmetric 2D float32 matrix with hermitian=True and tol
    M2 = np.random.randn(4, 4).astype(np.float32)
    M2 = M2 + M2.T
    list_of_inputs.append({
        "M": M2,
        "rtol": None,
        "hermitian": True,
        "tol": 1e-5
    })

    # Input 3: Basic 2D float64 matrix with rtol
    M3 = np.random.randn(6, 6).astype(np.float64)
    list_of_inputs.append({
        "M": M3,
        "rtol": 1e-6,
        "hermitian": False,
        "tol": None
    })

    # Input 4: Symmetric 2D float64 matrix with hermitian=True and tol
    M4 = np.random.randn(5, 5).astype(np.float64)
    M4 = M4 + M4.T
    list_of_inputs.append({
        "M": M4,
        "rtol": None,
        "hermitian": True,
        "tol": 1e-6
    })

    # Input 5: 3D tensor with rtol
    M5 = np.random.randn(2, 4, 4).astype(np.float32)
    list_of_inputs.append({
        "M": M5,
        "rtol": 1e-4,
        "hermitian": False,
        "tol": None
    })

    # Input 6: 3D tensor with tol
    M6 = np.random.randn(3, 5, 5).astype(np.float32)
    list_of_inputs.append({
        "M": M6,
        "rtol": None,
        "hermitian": False,
        "tol": 1e-4
    })

    # Input 7: Rectangular matrix (more columns) with rtol
    M7 = np.random.randn(3, 7).astype(np.float32)
    list_of_inputs.append({
        "M": M7,
        "rtol": 1e-5,
        "hermitian": False,
        "tol": None
    })

    # Input 8: Rectangular matrix (more rows) with tol
    M8 = np.random.randn(7, 3).astype(np.float32)
    list_of_inputs.append({
        "M": M8,
        "rtol": None,
        "hermitian": False,
        "tol": 1e-5
    })

    # Input 9: Rank deficient matrix with tol
    M9 = np.ones((5, 5), dtype=np.float32)
    list_of_inputs.append({
        "M": M9,
        "rtol": None,
        "hermitian": False,
        "tol": 1e-5
    })

    # Input 10: 4D tensor with rtol
    M10 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({
        "M": M10,
        "rtol": 1e-5,
        "hermitian": False,
        "tol": None
    })

    return list_of_inputs

generated_inputs["jax.numpy.linalg.matrix_rank_1"] = matrix_rank_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matrix_rank_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matrix_rank_1'.")


check_valid('jax.numpy.linalg.matrix_rank', generated_inputs['jax.numpy.linalg.matrix_rank_1'], lib="jax", suffix=1)
