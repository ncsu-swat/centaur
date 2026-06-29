
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def solve_sylvester_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 with "schur" method
    A = np.random.randn(2, 2).astype(np.float32)
    B = np.random.randn(2, 2).astype(np.float32)
    C = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        'A': A, 'B': B, 'C': C, 'method': 'schur', 'tol': 1e-08
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float32 with "eigen" method
    A = np.random.randn(2, 2).astype(np.float32)
    B = np.random.randn(2, 2).astype(np.float32)
    C = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        'A': A, 'B': B, 'C': C, 'method': 'eigen', 'tol': 1e-08
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, larger dimension, "schur"
    A = np.random.randn(5, 5).astype(np.float64)
    B = np.random.randn(4, 4).astype(np.float64)
    C = np.random.randn(5, 4).astype(np.float64)
    input_dict = {
        'A': A, 'B': B, 'C': C, 'method': 'schur', 'tol': 1e-12
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, larger dimension, "eigen"
    A = np.random.randn(5, 5).astype(np.float64)
    B = np.random.randn(4, 4).astype(np.float64)
    C = np.random.randn(5, 4).astype(np.float64)
    input_dict = {
        'A': A, 'B': B, 'C': C, 'method': 'eigen', 'tol': 1e-12
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched inputs (same batch size)
    A = np.random.randn(3, 2, 2).astype(np.float32)
    B = np.random.randn(3, 3, 3).astype(np.float32)
    C = np.random.randn(3, 2, 3).astype(np.float32)
    input_dict = {
        'A': A, 'B': B, 'C': C, 'method': 'eigen', 'tol': 1e-08
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batch broadcasting (A is 2D, B and C are 3D)
    A = np.random.randn(2, 2).astype(np.float32)
    B = np.random.randn(3, 3, 3).astype(np.float32)
    C = np.random.randn(3, 2, 3).astype(np.float32)
    input_dict = {
        'A': A, 'B': B, 'C': C, 'method': 'schur', 'tol': 1e-08
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch broadcasting (A has batch size 1, B has batch size 3)
    A = np.random.randn(1, 3, 3).astype(np.float32)
    B = np.random.randn(3, 2, 2).astype(np.float32)
    C = np.random.randn(3, 3, 2).astype(np.float32)
    input_dict = {
        'A': A, 'B': B, 'C': C, 'method': 'eigen', 'tol': 1e-08
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex inputs (complex64)
    A = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    B = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    C = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {
        'A': A, 'B': B, 'C': C, 'method': 'schur', 'tol': 1e-08
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex inputs (complex128), larger tol
    A = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex128)
    B = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex128)
    C = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex128)
    input_dict = {
        'A': A, 'B': B, 'C': C, 'method': 'eigen', 'tol': 1e-05
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High tolerance, small matrices with negative elements
    A = np.array([[-1.0, 0.5], [0.2, -2.0]], dtype=np.float32)
    B = np.array([[0.8, -0.1], [0.3, 1.2]], dtype=np.float32)
    C = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        'A': A, 'B': B, 'C': C, 'method': 'schur', 'tol': 1e-04
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.solve_sylvester"] = solve_sylvester_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.solve_sylvester' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.solve_sylvester'.")


check_valid('jax.scipy.linalg.solve_sylvester', generated_inputs['jax.scipy.linalg.solve_sylvester'], lib="jax", suffix=0)
