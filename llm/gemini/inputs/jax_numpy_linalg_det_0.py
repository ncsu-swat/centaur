
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def det_inputs():
    list_of_inputs = []

    # Input 1: Basic 2x2 float32 matrix
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 2: 3x3 float64 matrix with negative values
    a = np.array(
        [[-2.5, 1.2, 0.0], [3.1, -4.7, 8.2], [0.5, -1.1, -3.3]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 3: 1x1 float32 scalar matrix
    a = np.array([[5.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 4: 4x4 identity-like float32 matrix
    a = np.eye(4, dtype=np.float32) * -3.0
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 5: 2x2 complex64 matrix
    a = np.array([[1.0 + 2.0j, -1.0j], [3.0 - 0.5j, 4.0 + 1.0j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 6: 3x3 complex128 matrix
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 7: 3D batched matrices (shape: 5, 2, 2)
    a = np.random.randn(5, 2, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 8: 4D batched matrices (shape: 2, 3, 4, 4)
    a = np.random.randn(2, 3, 4, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 9: 5D batched matrices (shape: 2, 1, 2, 3, 3)
    a = np.random.randn(2, 1, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 10: 8x8 random float32 matrix
    a = np.random.randn(8, 8).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 11: Integer matrix (int32)
    a = np.array([[1, 2, 3], [0, -4, 5], [1, 0, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 12: Singular/zero matrix (shape: 3, 3)
    a = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.det"] = det_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.det' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.det'.")


check_valid('jax.numpy.linalg.det', generated_inputs['jax.numpy.linalg.det'], lib="jax", suffix=0)
