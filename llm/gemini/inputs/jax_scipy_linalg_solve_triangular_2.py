
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def solve_triangular_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D upper triangular, 1D vector b
    a = np.array([[3.0, 1.0, -2.0], [0.0, 2.0, 4.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": "N",
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Lower triangular, 2D matrix b
    a = np.array(
        [[2.0, 0.0, 0.0], [-1.0, 3.0, 0.0], [4.0, -2.0, 1.5]], dtype=np.float64
    )
    b = np.array([[1.0, -1.0], [2.0, 0.5], [-3.0, 4.0]], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "trans": "N",
        "lower": True,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Transposed 'T' with unit_diagonal=True
    a = np.array([[1.0, 2.0, -1.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    b = np.array([4.0, -2.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": "T",
        "lower": False,
        "unit_diagonal": True,
        "overwrite_b": True,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Conjugate transpose 'C' with complex numbers
    a = np.array(
        [
            [2.0 + 1.0j, 0.0, 0.0],
            [1.0 - 1.0j, 3.0 + 2.0j, 0.0],
            [-0.5 + 2.0j, 1.5 + 0.0j, -1.0 + 1.0j],
        ],
        dtype=np.complex64,
    )
    b = np.array([1.0 + 0.0j, -2.0 + 1.0j, 0.5 - 0.5j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "trans": "C",
        "lower": True,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched 3D upper triangular system
    np.random.seed(42)
    a = np.triu(np.random.randn(4, 5, 5)).astype(np.float32)
    for i in range(4):
        np.fill_diagonal(a[i], np.random.uniform(1.0, 3.0, size=5))
    b = np.random.randn(4, 5, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": "N",
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": True,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched lower triangular with trans='T'
    a = np.tril(np.random.randn(2, 3, 4, 4)).astype(np.float64)
    for idx in np.ndindex(2, 3):
        np.fill_diagonal(a[idx], np.random.uniform(1.0, 5.0, size=4))
    b = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "trans": "T",
        "lower": True,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1x1 scalar-like matrix
    a = np.array([[5.0]], dtype=np.float32)
    b = np.array([15.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": "N",
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex128 upper triangular with unit diagonal
    a = np.triu(
        np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    ).astype(np.complex128)
    b = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex128)
    input_dict = {
        "a": a,
        "b": b,
        "trans": "N",
        "lower": False,
        "unit_diagonal": True,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 2D lower triangular system with float64
    n = 16
    a = np.tril(np.random.randn(n, n)).astype(np.float64)
    np.fill_diagonal(a, np.random.uniform(2.0, 5.0, size=n))
    b = np.random.randn(n, 10).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "trans": "N",
        "lower": True,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batched complex conjugate transpose 'C' with unit_diagonal=True
    a = (np.random.randn(3, 4, 4) + 1j * np.random.randn(3, 4, 4)).astype(np.complex64)
    for i in range(3):
        a[i] = np.tril(a[i])
    b = (np.random.randn(3, 4, 2) + 1j * np.random.randn(3, 4, 2)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "trans": "C",
        "lower": True,
        "unit_diagonal": True,
        "overwrite_b": True,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.solve_triangular_2"] = solve_triangular_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.solve_triangular_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.solve_triangular_2'.")


check_valid('jax.scipy.linalg.solve_triangular', generated_inputs['jax.scipy.linalg.solve_triangular_2'], lib="jax", suffix=2)
