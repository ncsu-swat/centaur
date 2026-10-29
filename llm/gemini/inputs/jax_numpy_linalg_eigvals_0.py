
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def eigvals_inputs():
    list_of_inputs = []

    # Input 1: 2x2 float32 matrix
    a = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 2: 3x3 float32 matrix with negative values
    a = np.array(
        [[-2.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, -2.0]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 3: 4x4 float64 random matrix
    np.random.seed(42)
    a = np.random.randn(4, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 4: 1x1 float32 scalar-like matrix
    a = np.array([[42.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 5: Batch of matrices (3D), shape (2, 3, 3), float32
    a = np.random.randn(2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 6: Batch of matrices (3D), shape (5, 2, 2), float64
    a = np.random.randn(5, 2, 2).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 7: Multi-batch matrices (4D), shape (2, 3, 4, 4), float32
    a = np.random.randn(2, 3, 4, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 8: 2x2 complex64 matrix
    a = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 9: 3x3 complex128 matrix
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 10: Batch of complex64 matrices (3D), shape (4, 2, 2)
    a = (np.random.randn(4, 2, 2) + 1j * np.random.randn(4, 2, 2)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 11: 5x5 diagonal matrix with negative and positive values
    a = np.diag([-5.0, -1.0, 0.0, 2.0, 7.0]).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.eigvals"] = eigvals_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.eigvals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.eigvals'.")


check_valid('jax.numpy.linalg.eigvals', generated_inputs['jax.numpy.linalg.eigvals'], lib="jax", suffix=0)
