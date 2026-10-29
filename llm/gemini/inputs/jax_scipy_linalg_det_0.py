
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def det_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 float32 matrix
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": True}
    )

    # Input 2: 1x1 float64 matrix
    a = np.array([[5.5]], dtype=np.float64)
    list_of_inputs.append(
        {"a": a, "overwrite_a": True, "check_finite": False}
    )

    # Input 3: 3x3 matrix with negative and zero values
    a = np.array(
        [[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0], [0.0, -2.0, 1.0]],
        dtype=np.float32,
    )
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": False}
    )

    # Input 4: 4x4 identity matrix
    a = np.eye(4, dtype=np.float64)
    list_of_inputs.append(
        {"a": a, "overwrite_a": True, "check_finite": True}
    )

    # Input 5: Batched 3D matrix (batch size 2, 2x2)
    a = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[8.0, 5.0], [7.0, 9.0]]],
        dtype=np.float32,
    )
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": True}
    )

    # Input 6: Batched 4D matrix (2, 3, 2, 2)
    a = np.random.randn(2, 3, 2, 2).astype(np.float64)
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": False}
    )

    # Input 7: Complex matrix 2x2 (complex64)
    a = np.array(
        [[1.0 + 2.0j, 3.0 - 1.0j], [2.0 + 0.0j, -1.0 + 4.0j]],
        dtype=np.complex64,
    )
    list_of_inputs.append(
        {"a": a, "overwrite_a": True, "check_finite": True}
    )

    # Input 8: Complex matrix 3x3 (complex128)
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": False}
    )

    # Input 9: Large 10x10 matrix
    a = np.random.randn(10, 10).astype(np.float32)
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": True}
    )

    # Input 10: Batched 3D matrix (5, 4, 4)
    a = np.random.randn(5, 4, 4).astype(np.float32)
    list_of_inputs.append(
        {"a": a, "overwrite_a": True, "check_finite": False}
    )

    # Input 11: Singular 3x3 matrix (all zeros)
    a = np.zeros((3, 3), dtype=np.float64)
    list_of_inputs.append(
        {"a": a, "overwrite_a": False, "check_finite": True}
    )

    # Input 12: Diagonal 5x5 matrix
    a = np.diag(np.array([2.0, -3.0, 4.0, -1.0, 0.5], dtype=np.float32))
    list_of_inputs.append(
        {"a": a, "overwrite_a": True, "check_finite": True}
    )

    return list_of_inputs


generated_inputs["jax.scipy.linalg.det"] = det_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.det' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.det'.")


check_valid('jax.scipy.linalg.det', generated_inputs['jax.scipy.linalg.det'], lib="jax", suffix=0)
