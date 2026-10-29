
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def solve_triangular_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D upper triangular system with 1D RHS
    a = (
        np.triu(np.random.randn(4, 4).astype(np.float32))
        + np.eye(4, dtype=np.float32) * 2.0
    )
    b = np.random.randn(4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 0,
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Lower triangular system with 1D RHS
    a = (
        np.tril(np.random.randn(5, 5).astype(np.float64))
        + np.eye(5, dtype=np.float64) * 3.0
    )
    b = np.random.randn(5).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 0,
        "lower": True,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Upper triangular with transposed solve (trans=1) and 2D RHS
    a = (
        np.triu(np.random.randn(6, 6).astype(np.float32))
        + np.eye(6, dtype=np.float32) * 2.0
    )
    b = np.random.randn(6, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 1,
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": True,
        "debug": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Lower triangular with unit diagonal
    a = np.tril(np.random.randn(3, 3).astype(np.float32), k=-1)
    b = np.random.randn(3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 0,
        "lower": True,
        "unit_diagonal": True,
        "overwrite_b": False,
        "debug": True,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched 3D triangular matrices and 3D RHS
    batch_size, n, m = 3, 4, 2
    a = np.zeros((batch_size, n, n), dtype=np.float32)
    for i in range(batch_size):
        a[i] = (
            np.tril(np.random.randn(n, n).astype(np.float32))
            + np.eye(n, dtype=np.float32) * 2.5
        )
    b = np.random.randn(batch_size, n, m).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 0,
        "lower": True,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched 3D triangular matrices with 2D RHS
    batch_size, n = 4, 5
    a = np.zeros((batch_size, n, n), dtype=np.float64)
    for i in range(batch_size):
        a[i] = (
            np.triu(np.random.randn(n, n).astype(np.float64))
            + np.eye(n, dtype=np.float64) * 2.0
        )
    b = np.random.randn(batch_size, n).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 1,
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex64 upper triangular system with conjugate transpose (trans=2)
    real_part = (
        np.triu(np.random.randn(4, 4)) + np.eye(4, dtype=np.float32) * 2.0
    )
    imag_part = np.triu(np.random.randn(4, 4))
    a = (real_part + 1j * imag_part).astype(np.complex64)
    b = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 2,
        "lower": False,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex128 lower triangular system with 2D RHS
    real_part = (
        np.tril(np.random.randn(3, 3)) + np.eye(3, dtype=np.float64) * 3.0
    )
    imag_part = np.tril(np.random.randn(3, 3))
    a = (real_part + 1j * imag_part).astype(np.complex128)
    b = (np.random.randn(3, 2) + 1j * np.random.randn(3, 2)).astype(
        np.complex128
    )
    input_dict = {
        "a": a,
        "b": b,
        "trans": 2,
        "lower": True,
        "unit_diagonal": False,
        "overwrite_b": True,
        "debug": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D batched tensors
    shape_a = (2, 2, 4, 4)
    shape_b = (2, 2, 4, 3)
    a = np.zeros(shape_a, dtype=np.float32)
    for i in range(shape_a[0]):
        for j in range(shape_a[1]):
            a[i, j] = (
                np.tril(np.random.randn(4, 4).astype(np.float32))
                + np.eye(4, dtype=np.float32) * 2.0
            )
    b = np.random.randn(*shape_b).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 0,
        "lower": True,
        "unit_diagonal": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Unit diagonal upper triangular system with negative values
    a = np.triu(np.random.uniform(-5.0, -1.0, size=(5, 5)).astype(np.float32), k=1)
    b = np.random.uniform(-10.0, 10.0, size=(5, 2)).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "trans": 1,
        "lower": False,
        "unit_diagonal": True,
        "overwrite_b": False,
        "debug": True,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.solve_triangular_1"] = (
    solve_triangular_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.solve_triangular_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.solve_triangular_1'.")


check_valid('jax.scipy.linalg.solve_triangular', generated_inputs['jax.scipy.linalg.solve_triangular_1'], lib="jax", suffix=1)
