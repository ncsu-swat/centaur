
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def matrix_rank_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix, float32, full rank with rtol
    M = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "M": M,
        "rtol": 1e-5,
        "hermitian": False,
        "tol": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D square matrix, float64, rank-deficient with rtol
    M = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    input_dict = {
        "M": M,
        "rtol": 1e-4,
        "hermitian": False,
        "tol": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D rectangular matrix (tall), float32 with rtol
    M = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32
    )
    input_dict = {
        "M": M,
        "rtol": 1e-6,
        "hermitian": False,
        "tol": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D rectangular matrix (wide) with negative values with rtol
    M = np.array(
        [[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]], dtype=np.float32
    )
    input_dict = {
        "M": M,
        "rtol": 1e-3,
        "hermitian": False,
        "tol": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Real symmetric (Hermitian) matrix, hermitian=True with rtol
    A = np.random.randn(4, 4).astype(np.float32)
    M = A + A.T
    input_dict = {
        "M": M,
        "rtol": 1e-5,
        "hermitian": True,
        "tol": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D batched matrices, float32 with rtol
    M = np.random.randn(2, 3, 3).astype(np.float32)
    input_dict = {
        "M": M,
        "rtol": 1e-5,
        "hermitian": False,
        "tol": None,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex Hermitian matrix, hermitian=True with tol
    A = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    M = (A + A.conj().T).astype(np.complex64)
    input_dict = {
        "M": M,
        "rtol": None,
        "hermitian": True,
        "tol": 1e-4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D batched matrices, float64 with tol
    M = np.random.randn(2, 2, 4, 4).astype(np.float64)
    input_dict = {
        "M": M,
        "rtol": None,
        "hermitian": False,
        "tol": 1e-7,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All-zeros matrix (rank 0), float32 with tol
    M = np.zeros((3, 3), dtype=np.float32)
    input_dict = {
        "M": M,
        "rtol": None,
        "hermitian": False,
        "tol": 0.01,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1x1 matrix, float64 with tol
    M = np.array([[-5.5]], dtype=np.float64)
    input_dict = {
        "M": M,
        "rtol": None,
        "hermitian": False,
        "tol": 1e-5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
