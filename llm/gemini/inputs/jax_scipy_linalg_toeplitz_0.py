
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def toeplitz_inputs():
    list_of_inputs = []

    # Input 1: 1D standard float32 vectors, square result
    c = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    r = np.array([1.0, -1.0, -2.0, -3.0], dtype=np.float32)
    list_of_inputs.append({"c": c, "r": r})

    # Input 2: 1D rectangular (N != M), float64
    c = np.array([-5.0, 2.5, 0.0], dtype=np.float64)
    r = np.array([-5.0, 1.2, -3.4, 7.8, 9.1], dtype=np.float64)
    list_of_inputs.append({"c": c, "r": r})

    # Input 3: 1D integer arrays
    c = np.array([10, -20, 30, -40, 50], dtype=np.int32)
    r = np.array([10, 2, 3], dtype=np.int32)
    list_of_inputs.append({"c": c, "r": r})

    # Input 4: 1D complex numbers
    c = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    r = np.array([1.0 + 2.0j, 0.0 - 1.0j, 2.0 + 3.0j], dtype=np.complex64)
    list_of_inputs.append({"c": c, "r": r})

    # Input 5: 2D batched inputs with same shapes (B, N) and (B, M)
    c = np.random.randn(3, 5).astype(np.float32)
    r = np.random.randn(3, 5).astype(np.float32)
    list_of_inputs.append({"c": c, "r": r})

    # Input 6: 2D batched inputs with rectangular matrices (B, N) and (B, M)
    c = np.random.randn(4, 6).astype(np.float64)
    r = np.random.randn(4, 2).astype(np.float64)
    list_of_inputs.append({"c": c, "r": r})

    # Input 7: 2D broadcastable batch dimension (1, N) and (B, M)
    c = np.random.randn(1, 4).astype(np.float32)
    r = np.random.randn(5, 4).astype(np.float32)
    list_of_inputs.append({"c": c, "r": r})

    # Input 8: 3D multi-batch arrays (B1, B2, N) and (B1, B2, M)
    c = np.random.randn(2, 3, 4).astype(np.float32)
    r = np.random.randn(2, 3, 6).astype(np.float32)
    list_of_inputs.append({"c": c, "r": r})

    # Input 9: Single element arrays (1x1 output)
    c = np.array([42.0], dtype=np.float32)
    r = np.array([42.0], dtype=np.float32)
    list_of_inputs.append({"c": c, "r": r})

    # Input 10: 1D int64 array
    c = np.array([-1, 0, 1, 2, -3], dtype=np.int64)
    r = np.array([-1, 4, -5, 6], dtype=np.int64)
    list_of_inputs.append({"c": c, "r": r})

    # Input 11: 3D broadcastable shapes (B1, 1, N) and (1, B2, M)
    c = np.random.randn(3, 1, 5).astype(np.float64)
    r = np.random.randn(1, 4, 3).astype(np.float64)
    list_of_inputs.append({"c": c, "r": r})

    # Input 12: Complex128 batch array
    c = (np.random.randn(2, 4) + 1j * np.random.randn(2, 4)).astype(np.complex128)
    r = (np.random.randn(2, 4) + 1j * np.random.randn(2, 4)).astype(np.complex128)
    list_of_inputs.append({"c": c, "r": r})

    return list_of_inputs


generated_inputs["jax.scipy.linalg.toeplitz"] = toeplitz_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.toeplitz' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.toeplitz'.")


check_valid('jax.scipy.linalg.toeplitz', generated_inputs['jax.scipy.linalg.toeplitz'], lib="jax", suffix=0)
