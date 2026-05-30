
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tensordot_inputs():
    list_of_inputs = []

    # Input 1: float32, 3D and 3D arrays, default precision
    input_dict = {
        "a": np.random.randn(2, 3, 4).astype(np.float32),
        "b": np.random.randn(3, 4, 5).astype(np.float32),
        "axes": [[1, 2], [0, 1]],
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 2D and 2D arrays (matrix multiplication equivalent), high precision
    input_dict = {
        "a": np.random.randn(2, 2).astype(np.float32),
        "b": np.random.randn(2, 3).astype(np.float32),
        "axes": [[1], [0]],
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, 3D and 3D arrays, highest precision
    input_dict = {
        "a": np.random.randn(4, 3, 2).astype(np.float64),
        "b": np.random.randn(2, 3, 5).astype(np.float64),
        "axes": [[2, 1], [0, 1]],
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32 with negative values, summing over different axes
    input_dict = {
        "a": np.random.randint(-10, 10, size=(5, 5)).astype(np.int32),
        "b": np.random.randint(-10, 10, size=(5, 5)).astype(np.int32),
        "axes": [[0], [1]],
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, high-dimensional arrays, contracting 3 dimensions
    input_dict = {
        "a": np.random.randn(1, 2, 3, 4).astype(np.float32),
        "b": np.random.randn(4, 3, 2, 1).astype(np.float32),
        "axes": [[1, 2, 3], [2, 1, 0]],
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 2D arrays, high precision, returning float64 result
    input_dict = {
        "a": np.random.randn(3, 4).astype(np.float64),
        "b": np.random.randn(4, 2).astype(np.float64),
        "axes": [[1], [0]],
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int32, 1D arrays (dot product equivalent)
    input_dict = {
        "a": np.random.randint(-5, 5, size=(10,)).astype(np.int32),
        "b": np.random.randint(-5, 5, size=(10,)).astype(np.int32),
        "axes": [[0], [0]],
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, different dimensions, highest precision
    input_dict = {
        "a": np.random.randn(2, 3, 2).astype(np.float32),
        "b": np.random.randn(3, 2, 4).astype(np.float32),
        "axes": [[1, 2], [0, 1]],
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, 3D tensors, high precision
    input_dict = {
        "a": np.random.randn(3, 3, 3).astype(np.float32),
        "b": np.random.randn(3, 3, 3).astype(np.float32),
        "axes": [[2], [0]],
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, asymmetric shapes, default precision
    input_dict = {
        "a": np.random.randn(4, 5).astype(np.float64),
        "b": np.random.randn(5, 6).astype(np.float64),
        "axes": [[1], [0]],
        "precision": "default",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.tensordot_2"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tensordot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tensordot_2'.")


check_valid('jax.numpy.tensordot', generated_inputs['jax.numpy.tensordot_2'], lib="jax", suffix=2)
