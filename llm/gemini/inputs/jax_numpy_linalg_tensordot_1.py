
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tensordot_inputs():
    list_of_inputs = []

    # Input 1: Basic matrix multiplication equivalent (axes=1)
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 1,
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tensor contraction over 2 axes (axes=2)
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 2,
        "precision": "high",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Vector dot product (axes=1) with float64
    x1 = np.random.randn(5).astype(np.float64)
    x2 = np.random.randn(5).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 1,
        "precision": "highest",
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor outer product (axes=0)
    x1 = np.random.randn(2, 2).astype(np.float32)
    x2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 0,
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: High dimensional x1, low dimensional x2 (axes=1)
    x1 = np.random.randn(4, 3, 2).astype(np.float64)
    x2 = np.random.randn(2, 5).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 1,
        "precision": "high",
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer inputs with axes=2
    x1 = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    x2 = np.random.randint(-10, 10, size=(3, 4, 2)).astype(np.int32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 2,
        "precision": "default",
        "preferred_element_type": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element tensors contraction (axes=1)
    x1 = np.random.randn(1, 1, 1).astype(np.float32)
    x2 = np.random.randn(1, 1, 1).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 1,
        "precision": "highest",
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D and 3D tensors, contraction over last 1 axis of x1 and first of x2
    x1 = np.random.randn(3, 2, 4).astype(np.float32)
    x2 = np.random.randn(4, 2, 1).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 1,
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D and 3D tensors, contraction over 2 axes
    x1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    x2 = np.random.randn(2, 2, 3).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 2,
        "precision": "high",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative values and float32 outputs
    x1 = np.random.uniform(-5.0, 5.0, (5, 3)).astype(np.float32)
    x2 = np.random.uniform(-5.0, 5.0, (3, 2)).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axes": 1,
        "precision": "highest",
        "preferred_element_type": np.float32
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
