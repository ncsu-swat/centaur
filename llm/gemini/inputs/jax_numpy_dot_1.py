
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dot_inputs():
    list_of_inputs = []

    # Input 1: 1D dot 1D vectors with float32
    a = np.array([1.0, -2.5, 3.2], dtype=np.float32)
    b = np.array([-0.5, 4.0, 1.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D dot 2D matrix multiplication with float64
    a = np.random.randn(3, 4).astype(np.float64)
    b = np.random.randn(4, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D matrix dot 1D vector with negative integers
    a = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)
    b = np.array([2, -1, 0], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D vector dot 2D matrix
    a = np.array([1.5, -2.0, 0.5, 3.0], dtype=np.float32)
    b = np.random.randn(4, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D scalar arrays
    a = np.array(-3.5, dtype=np.float32)
    b = np.array(2.0, dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D scalar array dot 2D matrix
    a = np.array(4.0, dtype=np.float64)
    b = np.random.randn(3, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor dot 1D vector
    a = np.random.randn(2, 3, 5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor dot 2D matrix
    a = np.random.randn(2, 4, 3).astype(np.float32)
    b = np.random.randn(3, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor dot 3D tensor
    a = np.random.randn(2, 3, 4).astype(np.float64)
    b = np.random.randn(5, 4, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex dtypes with 2D matrices
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    b = (np.random.randn(3, 2) + 1j * np.random.randn(3, 2)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Boolean arrays accumulating to int32
    a = np.array([[True, False], [True, True]], dtype=np.bool_)
    b = np.array([[False, True], [True, False]], dtype=np.bool_)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32),
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
