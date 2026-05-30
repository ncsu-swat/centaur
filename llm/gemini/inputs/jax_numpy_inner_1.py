
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def inner_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D complex arrays with negative values
    a = np.array([1j, -3j, 4j], dtype=np.complex64)
    b = np.array([4.0, -2.0, 5.0], dtype=np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.dtype(np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, matching last dimension
    a = np.random.randn(2, 3).astype(np.float64)
    b = np.random.randn(5, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D and 1D arrays
    a = np.random.randn(4, 5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(1, 2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer inputs with a float preferred output type
    a = np.array([-1, 2, -3, 4], dtype=np.int32)
    b = np.array([5, -6, 7, -8], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Higher dimensional arrays (4D and 3D)
    a = np.random.randn(1, 2, 3, 5).astype(np.float64)
    b = np.random.randn(2, 1, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element vectors
    a = np.array([-10.5], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large last dimension
    a = np.random.randn(3, 100).astype(np.float64)
    b = np.random.randn(2, 100).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D and 1D arrays with complex inputs
    a = (np.random.randn(2, 2, 3) + 1j * np.random.randn(2, 2, 3)).astype(np.complex128)
    b = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex128)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.complex128)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.inner_1"] = inner_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.inner_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.inner_1'.")


check_valid('jax.numpy.inner', generated_inputs['jax.numpy.inner_1'], lib="jax", suffix=1)
