
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tensordot_inputs():
    list_of_inputs = []

    # Input 1: 3D and 3D tensors contracting two axes
    x1 = np.arange(24.0, dtype=np.float32).reshape(2, 3, 4) - 10.0
    x2 = np.ones((3, 4, 5), dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": ([1, 2], [0, 1]),
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D and 2D matrix multiplication
    x1 = np.random.randn(5, 7).astype(np.float32)
    x2 = np.random.randn(7, 4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": ([1], [0]),
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D vectors dot product
    x1 = np.array([-1.5, 2.0, -3.5, 4.0], dtype=np.float32)
    x2 = np.array([0.5, -2.0, 1.0, 3.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": ([0], [0]),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D and 3D tensors with mixed axes
    x1 = np.random.randn(2, 4, 3, 5).astype(np.float32)
    x2 = np.random.randn(5, 6, 4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": ([1, 3], [2, 0]),
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D and 3D tensor contraction
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": ([0], [1]),
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D and 4D contracting three axes
    x1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x2 = np.random.randn(4, 5, 2, 6).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": ([0, 2, 3], [2, 0, 1]),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Full Frobenius inner product of 2D matrices
    x1 = np.random.randn(4, 6).astype(np.float32)
    x2 = np.random.randn(4, 6).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": ([0, 1], [0, 1]),
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex numbers matrix contraction
    x1 = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex64)
    x2 = (np.random.randn(4, 2) + 1j * np.random.randn(4, 2)).astype(np.complex64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": ([1], [0]),
        "precision": "default",
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D and 2D tensor contraction
    x1 = np.random.randn(2, 2, 3, 2, 5).astype(np.float32)
    x2 = np.random.randn(5, 7).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": ([4], [0]),
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D and 2D contraction along multiple axes
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(4, 2).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": ([2, 0], [0, 1]),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.tensordot_2"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.tensordot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.tensordot_2'.")


check_valid('jax.numpy.linalg.tensordot', generated_inputs['jax.numpy.linalg.tensordot_2'], lib="jax", suffix=2)
