
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_numpy_linalg_matmul_inputs():
    list_of_inputs = []

    # Input 1: 1D @ 1D vector dot product
    x1 = np.array([1.5, -2.0, 3.0], dtype=np.float32)
    x2 = np.array([-0.5, 4.0, 1.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D @ 2D standard matrix multiplication with high precision and float64 accumulation
    x1 = np.random.randn(4, 5).astype(np.float32)
    x2 = np.random.randn(5, 3).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D @ 1D matrix-vector product
    x1 = np.random.randn(6, 4).astype(np.float64)
    x2 = np.random.randn(4).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D @ 2D vector-matrix product
    x1 = np.random.randint(-10, 10, size=(3,)).astype(np.int32)
    x2 = np.random.randint(-10, 10, size=(3, 5)).astype(np.int32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D @ 3D batched matrix multiplication
    x1 = np.random.randn(2, 4, 3).astype(np.float32)
    x2 = np.random.randn(2, 3, 5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D @ 4D with broadcast-compatible leading dimensions
    x1 = np.random.randn(2, 1, 5, 4).astype(np.float32)
    x2 = np.random.randn(1, 3, 4, 6).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex-valued matrix multiplication
    x1 = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex64
    )
    x2 = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex64
    )
    input_dict = {
        "x1": x1,
        "x2": x2,
        "precision": "default",
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batched 3D @ 2D broadcasting
    x1 = np.random.randn(4, 2, 8).astype(np.float32)
    x2 = np.random.randn(8, 3).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D @ 3D broadcasting
    x1 = np.random.randn(3, 7).astype(np.float64)
    x2 = np.random.randn(5, 7, 2).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D @ 1D batched matrix-vector multiplication
    x1 = np.random.randn(3, 6, 4).astype(np.float32)
    x2 = np.random.randn(4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.matmul_1"] = jax_numpy_linalg_matmul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matmul_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matmul_1'.")


check_valid('jax.numpy.linalg.matmul', generated_inputs['jax.numpy.linalg.matmul_1'], lib="jax", suffix=1)
