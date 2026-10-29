
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tensordot_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, axes=0 (outer product), default precision
    x1 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    x2 = np.array([-4.0, 5.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 0,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, axes=1 (vector dot product)
    x1 = np.array([1, -3, 2, 4], dtype=np.int32)
    x2 = np.array([5, 2, -1, 0], dtype=np.int32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 1,
        "precision": "high",
        "preferred_element_type": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, axes=1 (matrix multiplication)
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 1,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, axes=2 (matrix inner product/contraction of both dims)
    x1 = np.random.randn(3, 3).astype(np.float64)
    x2 = np.random.randn(3, 3).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 2,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays, axes=1
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(4, 5, 6).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 1,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays, axes=2
    x1 = np.random.randn(2, 4, 3).astype(np.float32)
    x2 = np.random.randn(4, 3, 5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 2,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays, axes=3 (complete contraction of 3D tensors)
    x1 = np.random.randn(2, 3, 4).astype(np.float64)
    x2 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 3,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D arrays, axes=2
    x1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x2 = np.random.randn(4, 5, 6, 7).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 2,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D arrays, axes=1
    x1 = np.random.randn(2, 2, 3, 4).astype(np.float32)
    x2 = np.random.randn(4, 5, 2, 2).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 1,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D integer arrays, axes=0 (outer product resulting in 4D tensor)
    x1 = np.arange(-6, 6, dtype=np.int32).reshape(3, 4)
    x2 = np.arange(1, 7, dtype=np.int32).reshape(2, 3)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 0,
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Complex valued arrays, axes=1
    x1 = (np.random.randn(3, 5) + 1j * np.random.randn(3, 5)).astype(
        np.complex64
    )
    x2 = (np.random.randn(5, 2) + 1j * np.random.randn(5, 2)).astype(
        np.complex64
    )
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 1,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.tensordot_1"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.tensordot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.tensordot_1'.")


check_valid('jax.numpy.linalg.tensordot', generated_inputs['jax.numpy.linalg.tensordot_1'], lib="jax", suffix=1)
