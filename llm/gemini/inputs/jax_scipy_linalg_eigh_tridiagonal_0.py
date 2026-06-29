
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def eigh_tridiagonal_inputs():
    list_of_inputs = []

    # Input 1: Small matrix, float32, all eigenvalues
    d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    e = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        'd': d,
        'e': e,
        'eigvals_only': True,
        'select': 'a',
        'select_range': (0.0, 0.0),
        'tol': 1e-12,
        'key': np.array([0, 0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Larger float64 matrix, all eigenvalues
    d = np.random.randn(10).astype(np.float64)
    e = np.random.randn(9).astype(np.float64)
    input_dict = {
        'd': d,
        'e': e,
        'eigvals_only': True,
        'select': 'a',
        'select_range': (0.0, 0.0),
        'tol': 0.0,
        'key': np.array([1, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Select index range, float32
    d = np.random.randn(5).astype(np.float32)
    e = np.random.randn(4).astype(np.float32)
    input_dict = {
        'd': d,
        'e': e,
        'eigvals_only': True,
        'select': 'i',
        'select_range': (0.0, 2.0),
        'tol': 1e-6,
        'key': np.array([42, 42], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values, N=8, float32
    d = np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0], dtype=np.float32)
    e = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    input_dict = {
        'd': d,
        'e': e,
        'eigvals_only': True,
        'select': 'a',
        'select_range': (0.0, 0.0),
        'tol': 1e-8,
        'key': np.array([10, 20], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: N=100, float64, select index range
    d = np.random.randn(100).astype(np.float64)
    e = np.random.randn(99).astype(np.float64)
    input_dict = {
        'd': d,
        'e': e,
        'eigvals_only': True,
        'select': 'i',
        'select_range': (10.0, 20.0),
        'tol': 1e-15,
        'key': np.array([100, 200], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Minimal size N=3, float32
    d = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    e = np.array([0.5, 0.5], dtype=np.float32)
    input_dict = {
        'd': d,
        'e': e,
        'eigvals_only': True,
        'select': 'a',
        'select_range': (0.0, 0.0),
        'tol': 1e-5,
        'key': np.array([9, 9], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: N=2, float32, select index range
    d = np.array([1.0, 2.0], dtype=np.float32)
    e = np.array([0.5], dtype=np.float32)
    input_dict = {
        'd': d,
        'e': e,
        'eigvals_only': True,
        'select': 'i',
        'select_range': (0.0, 1.0),
        'tol': 1e-7,
        'key': np.array([0, 1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: N=15, float64, select index range
    d = np.random.randn(15).astype(np.float64)
    e = np.random.randn(14).astype(np.float64)
    input_dict = {
        'd': d,
        'e': e,
        'eigvals_only': True,
        'select': 'i',
        'select_range': (5.0, 10.0),
        'tol': 1e-10,
        'key': np.array([5, 5], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large tolerance value, float32
    d = np.random.randn(6).astype(np.float32)
    e = np.random.randn(5).astype(np.float32)
    input_dict = {
        'd': d,
        'e': e,
        'eigvals_only': True,
        'select': 'a',
        'select_range': (0.0, 0.0),
        'tol': 1.0,
        'key': np.array([123, 456], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero off-diagonals (diagonal matrix), float32
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    e = np.zeros(6, dtype=np.float32)
    input_dict = {
        'd': d,
        'e': e,
        'eigvals_only': True,
        'select': 'i',
        'select_range': (1.0, 4.0),
        'tol': 1e-9,
        'key': np.array([7, 7], dtype=np.int32)
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
