
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def eigh_tridiagonal_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, select all
    n = 4
    d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    e = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "d": d,
        "e": e,
        "eigvals_only": True,
        "select": "a",
        "select_range": (0, n - 1),
        "tol": 1e-5,
        "key": np.array([0, 0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, select index subset
    n = 5
    d = np.array([2.5, -1.0, 4.2, -3.1, 0.5], dtype=np.float64)
    e = np.array([0.5, -0.2, 1.1, -0.4], dtype=np.float64)
    input_dict = {
        "d": d,
        "e": e,
        "eigvals_only": True,
        "select": "i",
        "select_range": (0, 2),
        "tol": 1e-6,
        "key": np.array([1, 2], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small 2x2 matrix, float32, zero tolerance
    n = 2
    d = np.array([-2.0, 3.0], dtype=np.float32)
    e = np.array([1.5], dtype=np.float32)
    input_dict = {
        "d": d,
        "e": e,
        "eigvals_only": True,
        "select": "a",
        "select_range": (0, 1),
        "tol": 0.0,
        "key": np.array([42, 42], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32, select single eigenvalue (smallest)
    n = 6
    d = np.linspace(-5.0, 5.0, n, dtype=np.float32)
    e = np.ones(n - 1, dtype=np.float32) * 0.5
    input_dict = {
        "d": d,
        "e": e,
        "eigvals_only": True,
        "select": "i",
        "select_range": (0, 0),
        "tol": 1e-5,
        "key": np.array([10, 20], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, select single eigenvalue (largest)
    n = 5
    d = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    e = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    input_dict = {
        "d": d,
        "e": e,
        "eigvals_only": True,
        "select": "i",
        "select_range": (4, 4),
        "tol": 1e-8,
        "key": np.array([99, 100], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Disconnected tridiagonal (zero off-diagonals)
    n = 4
    d = np.array([1.0, 4.0, 2.0, 3.0], dtype=np.float32)
    e = np.zeros(n - 1, dtype=np.float32)
    input_dict = {
        "d": d,
        "e": e,
        "eigvals_only": True,
        "select": "a",
        "select_range": (0, 3),
        "tol": 1e-4,
        "key": np.array([7, 13], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative diagonal elements
    n = 7
    d = -np.arange(1, n + 1, dtype=np.float64)
    e = np.full(n - 1, -0.5, dtype=np.float64)
    input_dict = {
        "d": d,
        "e": e,
        "eigvals_only": True,
        "select": "i",
        "select_range": (1, 4),
        "tol": 1e-7,
        "key": np.array([123, 456], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger matrix, float32, middle eigenvalues
    n = 10
    d = np.sin(np.arange(n, dtype=np.float32))
    e = np.cos(np.arange(n - 1, dtype=np.float32))
    input_dict = {
        "d": d,
        "e": e,
        "eigvals_only": True,
        "select": "i",
        "select_range": (3, 7),
        "tol": 1e-5,
        "key": np.array([2023, 2024], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values, float64, select all
    n = 3
    d = np.array([1e4, -2e4, 3e4], dtype=np.float64)
    e = np.array([1e2, -1e2], dtype=np.float64)
    input_dict = {
        "d": d,
        "e": e,
        "eigvals_only": True,
        "select": "a",
        "select_range": (0, 2),
        "tol": 1e-3,
        "key": np.array([5, 5], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Constant diagonal and off-diagonal (Toeplitz-like)
    n = 8
    d = np.full(n, 2.0, dtype=np.float32)
    e = np.full(n - 1, -1.0, dtype=np.float32)
    input_dict = {
        "d": d,
        "e": e,
        "eigvals_only": True,
        "select": "i",
        "select_range": (2, 5),
        "tol": 1e-6,
        "key": np.array([314, 159], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.eigh_tridiagonal"] = eigh_tridiagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.eigh_tridiagonal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.eigh_tridiagonal'.")


check_valid('jax.scipy.linalg.eigh_tridiagonal', generated_inputs['jax.scipy.linalg.eigh_tridiagonal'], lib="jax", suffix=0)
