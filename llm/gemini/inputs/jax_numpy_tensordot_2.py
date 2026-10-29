
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tensordot_inputs():
    list_of_inputs = []

    # Input 1: 2D arrays, 1 contraction axis (matrix multiplication)
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": [[1], [0]],
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D arrays, 2 contraction axes
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": [[1, 2], [0, 1]],
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D arrays with highest precision
    a = np.random.randn(4, 5, 2).astype(np.float32)
    b = np.random.randn(5, 2, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": [[1, 2], [0, 1]],
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D arrays (dot product)
    a = np.random.randn(10).astype(np.float32)
    b = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": [[0], [0]],
        "precision": "fastest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D tensors contracted along multiple non-consecutive axes
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    b = np.random.randn(3, 6, 5, 7).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": [[1, 3], [0, 2]],
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer arrays with int32 accumulation
    a = np.random.randint(-10, 10, size=(4, 6), dtype=np.int32)
    b = np.random.randint(-10, 10, size=(6, 3), dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": [[1], [0]],
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float16 inputs with float32 accumulation
    a = np.random.randn(3, 5).astype(np.float16)
    b = np.random.randn(5, 4).astype(np.float16)
    input_dict = {
        "a": a,
        "b": b,
        "axes": [[1], [0]],
        "precision": "float32",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative axis indexing
    a = np.random.randn(3, 4, 5).astype(np.float32)
    b = np.random.randn(2, 5, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": [[-2, -1], [-1, -2]],
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex numbers with complex64 dtype
    a = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(
        np.complex64
    )
    b = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "b": b,
        "axes": [[1], [0]],
        "precision": "high",
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D contraction using 1D list of axes
    a = np.random.randn(4, 5).astype(np.float32)
    b = np.random.randn(5, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": [1, 0],
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D tensors with single axis contraction
    a = np.random.randn(2, 2, 3, 2, 4).astype(np.float32)
    b = np.random.randn(4, 2, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": [[-1], [0]],
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.tensordot_2"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tensordot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tensordot_2'.")


check_valid('jax.numpy.tensordot', generated_inputs['jax.numpy.tensordot_2'], lib="jax", suffix=2)
