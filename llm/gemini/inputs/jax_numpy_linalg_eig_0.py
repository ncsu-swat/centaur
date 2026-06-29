
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_linalg_eig_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 square matrix (small)
    a = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 square matrix with negative values
    a = np.array([[-3.0, 1.0, 0.0], [1.0, -2.0, 4.0], [0.0, 4.0, -5.0]], dtype=np.float64)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D complex64 square matrix
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D complex128 square matrix
    a = (np.random.randn(5, 5) + 1j * np.random.randn(5, 5)).astype(np.complex128)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D batched float32 square matrices
    a = np.random.randn(2, 3, 3).astype(np.float32)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D batched float64 square matrices with negative values
    a = np.random.uniform(-10.0, 10.0, size=(4, 2, 2)).astype(np.float64)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D batched complex64 square matrices
    a = (np.random.randn(3, 4, 4) + 1j * np.random.randn(3, 4, 4)).astype(np.complex64)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D batched float32 square matrices
    a = np.random.randn(2, 2, 5, 5).astype(np.float32)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 square matrix of shape 1x1 (boundary case)
    a = np.array([[5.0]], dtype=np.float32)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 square matrix of larger dimension (10x10)
    a = np.random.randn(10, 10).astype(np.float64)
    input_dict = {"a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.eig"] = jax_numpy_linalg_eig_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.eig' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.eig'.")


check_valid('jax.numpy.linalg.eig', generated_inputs['jax.numpy.linalg.eig'], lib="jax", suffix=0)
