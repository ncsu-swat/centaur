
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def solve_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32, "gen" (vector b)
    a = np.random.randn(5, 5).astype(np.float32) + np.eye(5, dtype=np.float32) * 5
    b = np.random.randn(5).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'lower': False, 'overwrite_a': False, 'overwrite_b': False,
        'debug': False, 'check_finite': True, 'assume_a': 'gen'
    })

    # Input 2: Basic 2D float64, "gen", lower is True (vector b)
    a = np.random.randn(4, 4).astype(np.float64) + np.eye(4, dtype=np.float64) * 4
    b = np.random.randn(4).astype(np.float64)
    list_of_inputs.append({
        'a': a, 'b': b, 'lower': True, 'overwrite_a': False, 'overwrite_b': False,
        'debug': False, 'check_finite': True, 'assume_a': 'gen'
    })

    # Input 3: Symmetric matrix, "sym" (vector b)
    x = np.random.randn(6, 6).astype(np.float32)
    a = (x + x.T) + np.eye(6, dtype=np.float32) * 6
    b = np.random.randn(6).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'lower': False, 'overwrite_a': False, 'overwrite_b': False,
        'debug': False, 'check_finite': True, 'assume_a': 'sym'
    })

    # Input 4: Positive definite matrix, "pos" (vector b)
    x = np.random.randn(5, 5).astype(np.float32)
    a = np.dot(x, x.T) + np.eye(5, dtype=np.float32) * 2
    b = np.random.randn(5).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'lower': False, 'overwrite_a': False, 'overwrite_b': False,
        'debug': False, 'check_finite': True, 'assume_a': 'pos'
    })

    # Input 5: Hermitian complex64, "her" (vector b)
    x = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    a = (x + x.conj().T) + np.eye(4, dtype=np.complex64) * 4
    b = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    list_of_inputs.append({
        'a': a, 'b': b, 'lower': False, 'overwrite_a': False, 'overwrite_b': False,
        'debug': False, 'check_finite': True, 'assume_a': 'her'
    })

    # Input 6: Batched 3D float32, "gen" (b has shape (batch, N, M))
    a = np.random.randn(3, 5, 5).astype(np.float32) + np.eye(5, dtype=np.float32)[None, :, :] * 5
    b = np.random.randn(3, 5, 1).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'lower': False, 'overwrite_a': False, 'overwrite_b': False,
        'debug': False, 'check_finite': True, 'assume_a': 'gen'
    })

    # Input 7: Multi-column RHS (b has 2D shape), "gen"
    a = np.random.randn(6, 6).astype(np.float32) + np.eye(6, dtype=np.float32) * 6
    b = np.random.randn(6, 3).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'lower': False, 'overwrite_a': False, 'overwrite_b': False,
        'debug': False, 'check_finite': True, 'assume_a': 'gen'
    })

    # Input 8: Batched 3D positive definite, "pos" (b has shape (batch, N, M))
    x = np.random.randn(2, 4, 4).astype(np.float32)
    a = np.matmul(x, x.transpose(0, 2, 1)) + np.eye(4, dtype=np.float32)[None, :, :] * 2
    b = np.random.randn(2, 4, 2).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'lower': True, 'overwrite_a': False, 'overwrite_b': False,
        'debug': False, 'check_finite': True, 'assume_a': 'pos'
    })

    # Input 9: Large dimensions, float64, "gen" with overwritten/debug flags
    a = np.random.randn(128, 128).astype(np.float64) + np.eye(128, dtype=np.float64) * 128
    b = np.random.randn(128, 10).astype(np.float64)
    list_of_inputs.append({
        'a': a, 'b': b, 'lower': False, 'overwrite_a': True, 'overwrite_b': True,
        'debug': True, 'check_finite': False, 'assume_a': 'gen'
    })

    # Input 10: 4D Batched float32, "gen"
    a = np.random.randn(2, 2, 3, 3).astype(np.float32) + np.eye(3, dtype=np.float32)[None, None, :, :] * 5
    b = np.random.randn(2, 2, 3, 4).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'lower': False, 'overwrite_a': False, 'overwrite_b': False,
        'debug': False, 'check_finite': True, 'assume_a': 'gen'
    })

    return list_of_inputs

generated_inputs["jax.scipy.linalg.solve"] = solve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.solve'.")


check_valid('jax.scipy.linalg.solve', generated_inputs['jax.scipy.linalg.solve'], lib="jax", suffix=0)
