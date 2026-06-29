
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def det_inputs():
    list_of_inputs = []

    # Input 1: Small 2D square matrix (2x2), float32
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 2: 2D square matrix (3x3), float64 with negative values
    a = np.array([[-1.0, 2.0, 0.5], [3.0, -4.0, 1.5], [0.0, 1.0, -2.0]], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 3: Identity-like matrix (4x4), float32
    a = np.eye(4, dtype=np.float32) * 2.5
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 4: 3D batched square matrices (2, 3, 3), float32
    a = np.random.randn(2, 3, 3).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 5: 4D batched square matrices (2, 2, 4, 4), float64
    a = np.random.randn(2, 2, 4, 4).astype(np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 6: 2D complex matrix (2x2), complex64
    a = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 7: 2D complex matrix (3x3), complex128
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex128)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 8: Larger 2D square matrix (10x10), float32
    a = np.random.randn(10, 10).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 9: Singular matrix (determinant is 0) (3x3), float32
    a = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 10: 1x1 matrix, float32
    a = np.array([[5.5]], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 11: 5D batched square matrices (1, 2, 3, 2, 2), float32
    a = np.random.randn(1, 2, 3, 2, 2).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

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
