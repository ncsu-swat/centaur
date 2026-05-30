
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tensordot_inputs():
    list_of_inputs = []

    # Input 1
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 1,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 2,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.random.randn(5).astype(np.float64)
    b = np.random.randn(5).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 1,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    b = np.random.randn(3, 3, 4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 2,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 - including negative values
    a = np.random.uniform(-10.0, 10.0, size=(5, 2)).astype(np.float32)
    b = np.random.uniform(-10.0, 10.0, size=(2, 4)).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 1,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 - outer product with axes=0
    a = np.random.randn(3).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 0,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.random.randn(2, 2, 2).astype(np.float64)
    b = np.random.randn(2).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 1,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - integer type elements
    a = np.random.randint(-5, 5, size=(10, 10)).astype(np.int32)
    b = np.random.randint(-5, 5, size=(10, 5)).astype(np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 1,
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.random.randn(1, 1, 1).astype(np.float32)
    b = np.random.randn(1, 1, 1).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 1,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    b = np.random.randn(5, 6, 7, 8).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 1,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.tensordot_1"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tensordot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tensordot_1'.")


check_valid('jax.numpy.tensordot', generated_inputs['jax.numpy.tensordot_1'], lib="jax", suffix=1)
