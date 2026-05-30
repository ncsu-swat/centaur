
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vecmat_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D x1, 2D x2 (float32)
    x1 = np.random.randn(3).astype(np.float32)
    x2 = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: Simple 1D x1, 2D x2 with negative values (float64)
    x1 = np.random.uniform(-10.0, 10.0, size=(5,)).astype(np.float64)
    x2 = np.random.uniform(-10.0, 10.0, size=(5, 2)).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: Batched 2D x1, 3D x2 (float32)
    x1 = np.random.randn(2, 4).astype(np.float32)
    x2 = np.random.randn(2, 4, 3).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: Complex64 arrays (conjugate vecmat product testing)
    x1 = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    x2 = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: Broadcasted batch dimensions (x1 has batch size 1, x2 has batch size 4)
    x1 = np.random.randn(1, 3).astype(np.float32)
    x2 = np.random.randn(4, 3, 2).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: Broadcasted batch dimensions (x1 has batch size 4, x2 has batch size 1)
    x1 = np.random.randn(4, 3).astype(np.float32)
    x2 = np.random.randn(1, 3, 2).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: Multi-dimensional batching
    x1 = np.random.randn(2, 3, 5).astype(np.float32)
    x2 = np.random.randn(2, 3, 5, 4).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: Integer arrays (int32)
    x1 = np.random.randint(-5, 5, size=(4,)).astype(np.int32)
    x2 = np.random.randint(-5, 5, size=(4, 2)).astype(np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Large dimension arrays
    x1 = np.random.randn(128).astype(np.float32)
    x2 = np.random.randn(128, 256).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: Zero-valued array inputs
    x1 = np.zeros((2, 3), dtype=np.float32)
    x2 = np.zeros((2, 3, 3), dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 11: Complex128 arrays
    x1 = (np.random.randn(2) + 1j * np.random.randn(2)).astype(np.complex128)
    x2 = (np.random.randn(2, 5) + 1j * np.random.randn(2, 5)).astype(np.complex128)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.vecmat"] = vecmat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vecmat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vecmat'.")


check_valid('jax.numpy.vecmat', generated_inputs['jax.numpy.vecmat'], lib="jax", suffix=0)
