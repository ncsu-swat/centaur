
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def qr_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 square matrix, default mode 'full'
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 0,
        "mode": "full",
        "pivoting": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tall matrix (M > N), float64, mode 'economic'
    a = np.random.randn(5, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 1,
        "mode": "economic",
        "pivoting": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Wide matrix (M < N), float32, mode 'r'
    a = np.random.randn(3, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "overwrite_a": True,
        "lwork": 10,
        "mode": "r",
        "pivoting": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64 square matrix, mode 'full'
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 4,
        "mode": "full",
        "pivoting": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex128 matrix with negative components, mode 'economic'
    a = (np.random.randn(6, 4) - 2j * np.random.randn(6, 4)).astype(
        np.complex128
    )
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": -1,
        "mode": "economic",
        "pivoting": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D batched float32 matrices (batch_size, M, N)
    a = np.random.randn(2, 4, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 0,
        "mode": "full",
        "pivoting": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D batched float64 matrices
    a = np.random.randn(2, 3, 5, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "overwrite_a": True,
        "lwork": 2,
        "mode": "r",
        "pivoting": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Matrix with pivoting=True, mode 'full'
    a = np.random.randn(5, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 0,
        "mode": "full",
        "pivoting": True,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Matrix with pivoting=True, mode 'r'
    a = np.random.randn(4, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 5,
        "mode": "r",
        "pivoting": True,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Matrix with pivoting=True, mode 'economic'
    a = np.random.randn(6, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "overwrite_a": True,
        "lwork": 0,
        "mode": "economic",
        "pivoting": True,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D batched complex64 with pivoting=False, mode 'economic'
    a = (np.random.randn(3, 4, 2) + 1j * np.random.randn(3, 4, 2)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 0,
        "mode": "economic",
        "pivoting": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.qr"] = qr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.qr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.qr'.")


check_valid('jax.scipy.linalg.qr', generated_inputs['jax.scipy.linalg.qr'], lib="jax", suffix=0)
