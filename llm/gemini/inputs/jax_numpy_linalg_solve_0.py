
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def solve_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 2D solve, 1D RHS
    N = 3
    a = np.random.randn(N, N).astype(np.float32) + N * np.eye(N, dtype=np.float32)
    b = np.random.randn(N).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "b": copy.deepcopy(b)})

    # Input 2: float32 2D solve, 2D RHS (multiple systems)
    N, M = 4, 2
    a = np.random.randn(N, N).astype(np.float32) + N * np.eye(N, dtype=np.float32)
    b = np.random.randn(N, M).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "b": copy.deepcopy(b)})

    # Input 3: float64 precision
    N = 5
    a = np.random.randn(N, N).astype(np.float64) + N * np.eye(N, dtype=np.float64)
    b = np.random.randn(N).astype(np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a), "b": copy.deepcopy(b)})

    # Input 4: Batched solve (3D 'a', 3D 'b')
    B, N, M = 2, 3, 2
    a = np.random.randn(B, N, N).astype(np.float32) + N * np.eye(N, dtype=np.float32)[None, :, :]
    b = np.random.randn(B, N, M).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "b": copy.deepcopy(b)})

    # Input 5: Batched solve with 1D RHS 'b' (broadcasting)
    B, N = 3, 4
    a = np.random.randn(B, N, N).astype(np.float32) + N * np.eye(N, dtype=np.float32)[None, :, :]
    b = np.random.randn(N).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "b": copy.deepcopy(b)})

    # Input 6: Complex numbers
    N = 3
    a = (np.random.randn(N, N) + 1j * np.random.randn(N, N)).astype(np.complex64) + N * np.eye(N, dtype=np.complex64)
    b = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a), "b": copy.deepcopy(b)})

    # Input 7: Large dimensions
    N = 64
    a = np.random.randn(N, N).astype(np.float32) + N * np.eye(N, dtype=np.float32)
    b = np.random.randn(N).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "b": copy.deepcopy(b)})

    # Input 8: Multi-dimensional batch (4D 'a', 4D 'b')
    B1, B2, N, M = 2, 2, 3, 2
    a = np.random.randn(B1, B2, N, N).astype(np.float32) + N * np.eye(N, dtype=np.float32)[None, None, :, :]
    b = np.random.randn(B1, B2, N, M).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "b": copy.deepcopy(b)})

    # Input 9: Negative values, float32
    N = 4
    a = -np.random.uniform(1, 5, size=(N, N)).astype(np.float32) - N * np.eye(N, dtype=np.float32)
    b = np.random.uniform(-10, -1, size=(N,)).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "b": copy.deepcopy(b)})

    # Input 10: 2x2 simple system, float32
    N, M = 2, 1
    a = np.random.randn(N, N).astype(np.float32) + N * np.eye(N, dtype=np.float32)
    b = np.random.randn(N, M).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "b": copy.deepcopy(b)})

    return list_of_inputs

generated_inputs["jax.numpy.linalg.solve"] = solve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.solve'.")


check_valid('jax.numpy.linalg.solve', generated_inputs['jax.numpy.linalg.solve'], lib="jax", suffix=0)
