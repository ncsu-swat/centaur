
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def eig_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 float32 square matrix
    a = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 2: 3x3 float64 matrix with negative values
    a = np.array(
        [[-1.5, 2.0, 0.5], [0.0, -3.0, 4.0], [1.0, -2.0, 2.5]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 3: 4x4 identity matrix (float32)
    a = np.eye(4, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 4: Complex128 2x2 matrix
    a = np.array(
        [[1.0 + 2.0j, -1.0j], [2.0 - 0.5j, 3.0 + 1.0j]], dtype=np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 5: Complex64 3x3 matrix
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 6: Batch of 2x2 matrices, 3D shape (5, 2, 2) float32
    a = np.random.randn(5, 2, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 7: Multi-batch 4D shape (2, 3, 4, 4) float64
    a = np.random.randn(2, 3, 4, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 8: Batch of complex64 matrices (4, 3, 3)
    a = (np.random.randn(4, 3, 3) + 1j * np.random.randn(4, 3, 3)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 9: 1x1 matrix float32
    a = np.array([[-4.2]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 10: 5x5 random symmetric matrix float32
    sym = np.random.randn(5, 5).astype(np.float32)
    a = sym + sym.T
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 11: 3-dimensional batch of 1x1 matrices (3, 2, 1, 1) float64
    a = np.random.randn(3, 2, 1, 1).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.eig"] = eig_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.eig' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.eig'.")


check_valid('jax.numpy.linalg.eig', generated_inputs['jax.numpy.linalg.eig'], lib="jax", suffix=0)
