
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def eigvals_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 float32 matrix
    a = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 2: 3x3 float64 matrix with random values
    a = np.random.randn(3, 3).astype(np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 3: Batched 2x2 matrices, shape (3, 2, 2) float32
    a = np.random.randn(3, 2, 2).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 4: Batched 3x3 matrices, shape (2, 4, 3, 3) float64
    a = np.random.randn(2, 4, 3, 3).astype(np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 5: Edge case - 1x1 matrix float32
    a = np.array([[42.0]], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 6: 3x3 Complex64 matrix
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 7: 4x4 Complex128 matrix
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex128)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 8: 5x5 Identity matrix float32
    a = np.eye(5, dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 9: 6x6 upper triangular matrix float32
    a = np.triu(np.random.randn(6, 6)).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 10: 10x10 random symmetric matrix float64
    base = np.random.randn(10, 10).astype(np.float64)
    a = base + base.T
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 11: 8x8 zero matrix float32
    a = np.zeros((8, 8), dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

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
