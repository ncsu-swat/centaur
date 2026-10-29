
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def matrix_norm_inputs():
    list_of_inputs = []

    # Input 1: 2D square float32 matrix, frobenius norm
    x = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D square float64 matrix, nuclear norm, keepdims=True
    x = np.array([[2.0, 0.0], [0.0, -3.0]], dtype=np.float64)
    input_dict = {"x": x, "keepdims": True, "ord": "nuc"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D rectangular (wide) matrix, frobenius norm
    x = np.array(
        [[-1.5, 2.5, 3.0, -0.5], [0.0, 1.2, -4.1, 2.3]], dtype=np.float32
    )
    input_dict = {"x": x, "keepdims": False, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D rectangular (tall) matrix, nuclear norm
    x = np.array([[1.0, 2.0], [-3.0, 4.0], [5.0, -6.0]], dtype=np.float32)
    input_dict = {"x": x, "keepdims": True, "ord": "nuc"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D batched matrices, frobenius norm, keepdims=False
    x = np.random.randn(4, 3, 3).astype(np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D batched matrices, nuclear norm, keepdims=True
    x = np.random.randn(2, 4, 5).astype(np.float64)
    input_dict = {"x": x, "keepdims": True, "ord": "nuc"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D batched matrices, frobenius norm
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex-valued 2D matrix, frobenius norm
    x = (
        np.array([[1.0 + 2.0j, -3.0 + 0.5j], [0.0 - 1.0j, 4.0 + 2.5j]])
    ).astype(np.complex64)
    input_dict = {"x": x, "keepdims": True, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex-valued 2D matrix, nuclear norm
    x = (
        np.array([[2.0 - 1.0j, 1.0 + 0.0j], [-1.0 + 3.0j, 0.5 - 2.0j]])
    ).astype(np.complex128)
    input_dict = {"x": x, "keepdims": False, "ord": "nuc"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D batched matrices with negative values, keepdims=True, frobenius norm
    x = -np.abs(np.random.randn(3, 2, 5, 3)).astype(np.float64)
    input_dict = {"x": x, "keepdims": True, "ord": "fro"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D batched tall matrices, nuclear norm, keepdims=False
    x = np.random.randn(5, 6, 2).astype(np.float32)
    input_dict = {"x": x, "keepdims": False, "ord": "nuc"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.matrix_norm_1"] = matrix_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matrix_norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matrix_norm_1'.")


check_valid('jax.numpy.linalg.matrix_norm', generated_inputs['jax.numpy.linalg.matrix_norm_1'], lib="jax", suffix=1)
