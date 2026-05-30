
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def inner_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    a = np.random.randn(5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 arrays, high precision
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values, float64, highest precision
    a = np.random.uniform(-10, 10, size=(2, 3)).astype(np.float64)
    b = np.random.uniform(-10, 10, size=(4, 3)).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": "float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher dimensions (3D)
    a = np.random.randn(2, 3, 5).astype(np.float32)
    b = np.random.randn(4, 2, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integer arrays with int32 preferred type
    a = np.random.randint(-5, 5, size=(6,)).astype(np.int32)
    b = np.random.randint(-5, 5, size=(6,)).astype(np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "int32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Mixed dimensions (3D and 1D)
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed dimensions (1D and 2D), float64
    a = np.random.randn(6).astype(np.float64)
    b = np.random.randn(3, 6).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": "float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger dimensions
    a = np.random.randn(50, 10).astype(np.float32)
    b = np.random.randn(25, 10).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": "float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D arrays, float32 precision and float64 output
    a = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(2, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": "float64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 with high precision
    a = np.random.randn(12).astype(np.float64)
    b = np.random.randn(12).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": "float64"
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
