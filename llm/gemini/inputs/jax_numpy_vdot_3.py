
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vdot_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays, default precision
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 arrays with negative values, high precision
    a = np.array([-1.5, 2.5, -3.5, 4.5], dtype=np.float64)
    b = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": "float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex64 arrays, highest precision
    a = np.array([1 + 1j, 2 - 2j, 3 + 3j], dtype=np.complex64)
    b = np.array([1 - 1j, 2 + 2j, 3 - 3j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": "complex64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multidimensional inputs (2D and 1D) that flatten to same size
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(6).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D square matrices (will be flattened), high precision
    a = np.random.randn(3, 3).astype(np.float64)
    b = np.random.randn(3, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": "float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer inputs, returning float32 preferred type
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([5, 6, 7, 8], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D and 1D inputs, returning float64 preferred type
    a = np.random.randn(2, 2, 2).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single-element arrays
    a = np.array([1.0], dtype=np.float32)
    b = np.array([-1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex128 arrays, highest precision
    a = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex128)
    b = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex128)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": "complex128"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D arrays of different shapes that flatten to the same size
    a = np.random.randn(1, 10).astype(np.float32)
    b = np.random.randn(10, 1).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.vdot_3"] = vdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vdot_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vdot_3'.")


check_valid('jax.numpy.vdot', generated_inputs['jax.numpy.vdot_3'], lib="jax", suffix=3)
