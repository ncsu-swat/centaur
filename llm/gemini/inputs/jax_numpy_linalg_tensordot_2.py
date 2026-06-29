
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tensordot_inputs():
    list_of_inputs = []

    # Input 1: 2D arrays, contracting on 1 axis
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(4, 5).astype(np.float32)
    list_of_inputs.append({
        "x1": x1,
        "x2": x2,
        "axes": ((1,), (0,)),
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 2: 3D arrays contracting on 2 axes, with negative values
    x1 = np.random.uniform(-10, 10, (2, 3, 4)).astype(np.float32)
    x2 = np.random.uniform(-10, 10, (3, 4, 5)).astype(np.float32)
    list_of_inputs.append({
        "x1": x1,
        "x2": x2,
        "axes": ((1, 2), (0, 1)),
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 3: 1D arrays (dot product)
    x1 = np.random.randn(5).astype(np.float32)
    x2 = np.random.randn(5).astype(np.float32)
    list_of_inputs.append({
        "x1": x1,
        "x2": x2,
        "axes": ((0,), (0,)),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 4: 4D and 3D arrays contracting on 2 axes
    x1 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    x2 = np.random.randn(3, 3, 4).astype(np.float32)
    list_of_inputs.append({
        "x1": x1,
        "x2": x2,
        "axes": ((2, 3), (0, 1)),
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 5: 2D arrays, fully contracted (scalar result)
    x1 = np.random.randn(3, 3).astype(np.float32)
    x2 = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({
        "x1": x1,
        "x2": x2,
        "axes": ((0, 1), (0, 1)),
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 6: Integer arrays
    x1 = np.random.randint(-5, 5, (2, 3)).astype(np.int32)
    x2 = np.random.randint(-5, 5, (3, 2)).astype(np.int32)
    list_of_inputs.append({
        "x1": x1,
        "x2": x2,
        "axes": ((1,), (0,)),
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32)
    })

    # Input 7: 3D and 4D with complex axis mapping
    x1 = np.random.randn(3, 4, 5).astype(np.float32)
    x2 = np.random.randn(2, 5, 3, 4).astype(np.float32)
    list_of_inputs.append({
        "x1": x1,
        "x2": x2,
        "axes": ((0, 1, 2), (2, 3, 1)),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 8: 1D and 2D arrays
    x1 = np.random.randn(3).astype(np.float32)
    x2 = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({
        "x1": x1,
        "x2": x2,
        "axes": ((0,), (0,)),
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 9: Large dimensions, float32
    x1 = np.random.randn(10, 10).astype(np.float32)
    x2 = np.random.randn(10, 10).astype(np.float32)
    list_of_inputs.append({
        "x1": x1,
        "x2": x2,
        "axes": ((1,), (0,)),
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    })

    # Input 10: 4D arrays contracting on 1 axis
    x1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x2 = np.random.randn(5, 4, 3, 2).astype(np.float32)
    list_of_inputs.append({
        "x1": x1,
        "x2": x2,
        "axes": ((3,), (0,)),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32)
    })

    return list_of_inputs

generated_inputs["jax.numpy.linalg.tensordot_2"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.tensordot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.tensordot_2'.")


check_valid('jax.numpy.linalg.tensordot', generated_inputs['jax.numpy.linalg.tensordot_2'], lib="jax", suffix=2)
