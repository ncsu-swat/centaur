
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tensordot_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D and 3D tensor dot with float32
    a = np.random.randn(3, 4, 5).astype(np.float32)
    b = np.random.randn(4, 5, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": ((1, 2), (0, 1)),
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D matrices (equivalent to matmul) with float32 (avoid float64 warning)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": ((1,), (0,)),
        "precision": "high",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer tensors
    a = np.random.randint(-10, 10, size=(5, 5)).astype(np.int32)
    b = np.random.randint(-10, 10, size=(5, 5)).astype(np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": ((1,), (0,)),
        "precision": "default",
        "preferred_element_type": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Total contraction of 3D tensors with highest precision
    a = np.random.randn(2, 2, 2).astype(np.float32)
    b = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": ((0, 1, 2), (0, 1, 2)),
        "precision": "highest",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float16 precision inputs, accumulating in Float32
    a = np.random.randn(4, 3, 2).astype(np.float16)
    b = np.random.randn(2, 3, 5).astype(np.float16)
    input_dict = {
        "a": a,
        "b": b,
        "axes": ((2, 1), (0, 1)),
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensors contraction along non-adjacent axes
    a = np.random.randn(3, 3, 3).astype(np.float32)
    b = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": ((2, 0), (1, 2)),
        "precision": "high",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Contraction with negative values and varying shapes
    a = np.random.uniform(-5.0, 5.0, (1, 5, 2)).astype(np.float32)
    b = np.random.uniform(-5.0, 5.0, (2, 5, 3)).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": ((2, 1), (0, 1)),
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Int32 inputs, accumulating as Int32
    a = np.random.randint(-5, 5, size=(4, 2)).astype(np.int32)
    b = np.random.randint(-5, 5, size=(2, 4)).astype(np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": ((1,), (0,)),
        "precision": "default",
        "preferred_element_type": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large dimensions contraction
    a = np.random.randn(10, 10, 2).astype(np.float32)
    b = np.random.randn(10, 2, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": ((0, 2), (0, 1)),
        "precision": "highest",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex axis mapping
    a = np.random.randn(6, 7, 8).astype(np.float32)
    b = np.random.randn(8, 7, 9).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": ((1, 2), (1, 0)),
        "precision": "high",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.tensordot_3"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tensordot_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tensordot_3'.")


check_valid('jax.numpy.tensordot', generated_inputs['jax.numpy.tensordot_3'], lib="jax", suffix=3)
