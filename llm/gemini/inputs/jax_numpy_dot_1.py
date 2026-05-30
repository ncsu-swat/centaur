
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dot_inputs():
    list_of_inputs = []

    # Input 1: 1D vector dot product (float32)
    a = np.random.randn(10).astype(np.float32)
    b = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D matrix multiplication (float32, high precision)
    a = np.random.randn(5, 8).astype(np.float32)
    b = np.random.randn(8, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Matrix-vector multiplication (float64, highest precision)
    a = np.random.randn(4, 7).astype(np.float64)
    b = np.random.randn(7).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Vector-matrix multiplication (int32)
    a = np.random.randint(-10, 10, size=(6,)).astype(np.int32)
    b = np.random.randint(-10, 10, size=(6, 4)).astype(np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D stacked matrices (float32, custom accumulation type float64)
    a = np.random.randn(2, 3, 5).astype(np.float32)
    b = np.random.randn(2, 5, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative and large scale values (float64)
    a = np.random.uniform(-100, 100, size=(10, 10)).astype(np.float64)
    b = np.random.uniform(-100, 100, size=(10, 10)).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex matrix multiplication (complex64)
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    b = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.complex128
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High-dimensional stacking (float32)
    a = np.random.randn(2, 2, 4).astype(np.float32)
    b = np.random.randn(2, 4, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-square dimensions (float32, highest precision)
    a = np.random.randn(12, 1).astype(np.float32)
    b = np.random.randn(1, 15).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger matrices for performance benchmarking (float32, default precision)
    a = np.random.randn(128, 64).astype(np.float32)
    b = np.random.randn(64, 128).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.dot_1"] = dot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dot_1'.")


check_valid('jax.numpy.dot', generated_inputs['jax.numpy.dot_1'], lib="jax", suffix=1)
