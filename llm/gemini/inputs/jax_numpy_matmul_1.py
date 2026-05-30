
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def matmul_inputs():
    list_of_inputs = []

    # Input 1: Standard 2D float32
    list_of_inputs.append({
        "a": np.random.randn(5, 10).astype(np.float32),
        "b": np.random.randn(10, 3).astype(np.float32),
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 2: 1D Dot product
    list_of_inputs.append({
        "a": np.random.randn(10).astype(np.float32),
        "b": np.random.randn(10).astype(np.float32),
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 3: 3D Batched multiplication
    list_of_inputs.append({
        "a": np.random.randn(2, 4, 8).astype(np.float32),
        "b": np.random.randn(2, 8, 3).astype(np.float32),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 4: Double precision float64
    list_of_inputs.append({
        "a": np.random.randn(4, 4).astype(np.float64),
        "b": np.random.randn(4, 4).astype(np.float64),
        "precision": "default",
        "preferred_element_type": np.dtype(np.float64)
    })

    # Input 5: Integers with negative values
    list_of_inputs.append({
        "a": np.random.randint(-10, 10, size=(3, 5)).astype(np.int32),
        "b": np.random.randint(-10, 10, size=(5, 4)).astype(np.int32),
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32)
    })

    # Input 6: Vector-Matrix product (1D a, 2D b)
    list_of_inputs.append({
        "a": np.random.randn(6).astype(np.float32),
        "b": np.random.randn(6, 4).astype(np.float32),
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 7: Matrix-Vector product (2D a, 1D b)
    list_of_inputs.append({
        "a": np.random.randn(4, 7).astype(np.float32),
        "b": np.random.randn(7).astype(np.float32),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 8: Multi-dimensional batches (4D)
    list_of_inputs.append({
        "a": np.random.randn(2, 3, 5, 6).astype(np.float32),
        "b": np.random.randn(2, 3, 6, 4).astype(np.float32),
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 9: Complex inputs
    list_of_inputs.append({
        "a": (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64),
        "b": (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64),
        "precision": "high",
        "preferred_element_type": np.dtype(np.complex64)
    })

    # Input 10: Large matrix size float32
    list_of_inputs.append({
        "a": np.random.randn(128, 256).astype(np.float32),
        "b": np.random.randn(256, 128).astype(np.float32),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32)
    })

    return list_of_inputs

generated_inputs["jax.numpy.matmul_1"] = matmul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.matmul_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.matmul_1'.")


check_valid('jax.numpy.matmul', generated_inputs['jax.numpy.matmul_1'], lib="jax", suffix=1)
