
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def inner_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 vectors
    a = np.array([1.0, -2.0, 3.5, 4.0], dtype=np.float32)
    b = np.array([-0.5, 2.0, 1.0, -3.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 vectors with high precision
    a = np.array([1.234, -5.678, 9.012], dtype=np.float64)
    b = np.array([-3.141, 2.718, 0.577], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": "float64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer matrices
    a = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    b = np.array([[7, 8, 9], [-1, 0, 2], [3, -4, 5]], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": "int32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D matrix and 1D vector
    a = np.random.randn(4, 6).astype(np.float32)
    b = np.random.randn(6).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D vector and 2D matrix, upcasting to float64
    a = np.random.randn(5).astype(np.float32)
    b = np.random.randn(3, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": "float64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor and 2D matrix
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(5, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex-valued 3D tensors
    a = (np.random.randn(2, 2, 3) + 1j * np.random.randn(2, 2, 3)).astype(
        np.complex64
    )
    b = (np.random.randn(4, 1, 3) + 1j * np.random.randn(4, 1, 3)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "complex64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D matrices with single row
    a = np.array([[-1.5, 2.5, -3.5, 4.5]], dtype=np.float32)
    b = np.array([[0.5, -1.0, 2.0, -0.5]], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": "float32",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensor and 2D tensor with int64
    a = np.random.randint(-10, 10, size=(2, 1, 3, 4)).astype(np.int64)
    b = np.random.randint(-10, 10, size=(2, 4)).astype(np.int64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "int64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D and 3D tensors with float64
    a = np.random.randn(3, 5).astype(np.float64)
    b = np.random.randn(2, 4, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": "float64",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.inner_3"] = inner_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.inner_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.inner_3'.")


check_valid('jax.numpy.inner', generated_inputs['jax.numpy.inner_3'], lib="jax", suffix=3)
