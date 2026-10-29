
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def matmul_inputs():
    list_of_inputs = []

    # Input 1: 1D @ 1D vectors, float32, default precision
    input_dict = {
        "a": np.array([1.0, -2.0, 3.5], dtype=np.float32),
        "b": np.array([-4.0, 5.0, 6.0], dtype=np.float32),
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D @ 2D matrices, int32, high precision
    input_dict = {
        "a": np.array([[1, -2], [3, 4]], dtype=np.int32),
        "b": np.array([[5, 6], [-7, 8]], dtype=np.int32),
        "precision": "high",
        "preferred_element_type": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D @ 1D matrix-vector product, float64, highest precision
    input_dict = {
        "a": np.random.randn(5, 4).astype(np.float64),
        "b": np.random.randn(4).astype(np.float64),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D @ 2D vector-matrix product, float32, default precision
    input_dict = {
        "a": np.random.randn(3).astype(np.float32),
        "b": np.random.randn(3, 6).astype(np.float32),
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched 3D @ 3D matrices, float32, high precision
    input_dict = {
        "a": np.random.randn(8, 4, 3).astype(np.float32),
        "b": np.random.randn(8, 3, 5).astype(np.float32),
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasted batch dimensions (1, 4, 2) @ (3, 2, 5), float64, highest precision
    input_dict = {
        "a": np.random.randn(1, 4, 2).astype(np.float64),
        "b": np.random.randn(3, 2, 5).astype(np.float64),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex numbers 2D @ 2D, complex64, default precision
    input_dict = {
        "a": (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
            np.complex64
        ),
        "b": (np.random.randn(3, 2) + 1j * np.random.randn(3, 2)).astype(
            np.complex64
        ),
        "precision": "default",
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensors batch matmul, float32, high precision
    input_dict = {
        "a": np.random.randn(2, 3, 5, 4).astype(np.float32),
        "b": np.random.randn(2, 3, 4, 7).astype(np.float32),
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-square rectangular matrices, accumulation to float64, highest precision
    input_dict = {
        "a": np.random.randn(10, 2).astype(np.float32),
        "b": np.random.randn(2, 8).astype(np.float32),
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D broadcast with 2D matrix, float32, default precision
    input_dict = {
        "a": np.random.randn(2, 4, 6, 8).astype(np.float32),
        "b": np.random.randn(8, 3).astype(np.float32),
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.matmul_1"] = matmul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.matmul_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.matmul_1'.")


check_valid('jax.numpy.matmul', generated_inputs['jax.numpy.matmul_1'], lib="jax", suffix=1)
