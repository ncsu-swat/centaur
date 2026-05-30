
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vdot_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D real float arrays
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex arrays to demonstrate conjugate multiplication
    a = np.array([1j, 2j, 3j], dtype=np.complex64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float values
    a = np.array([-1.5, -2.5, -3.5], dtype=np.float32)
    b = np.array([2.0, -3.0, 4.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays that will be flattened (same shape)
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays with different shapes but same size (will be flattened)
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32) 
    b = np.array([[4.0], [5.0], [6.0]], dtype=np.float32) 
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays
    a = np.ones((2, 2, 2), dtype=np.float32)
    b = np.ones((2, 2, 2), dtype=np.float32) * 2.0
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean arrays
    a = np.array([True, False, True], dtype=bool)
    b = np.array([True, True, False], dtype=bool)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Double precision complex numbers
    a = np.array([1.0 + 1j, 2.0 + 2j], dtype=np.complex128)
    b = np.array([3.0 - 3j, 4.0 - 4j], dtype=np.complex128)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.complex128
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D array with mixed integers/floats
    a = np.arange(100, dtype=np.float32)
    b = np.arange(100, dtype=np.float32) * -0.5
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single-element tensors
    a = np.array([5.5], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.vdot_1"] = vdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vdot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vdot_1'.")


check_valid('jax.numpy.vdot', generated_inputs['jax.numpy.vdot_1'], lib="jax", suffix=1)
