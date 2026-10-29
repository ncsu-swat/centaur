
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def matrix_transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 square matrix
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D int32 rectangular matrix with negative values
    x = np.array([[-1, -2, -3], [4, 5, -6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 3D float64 batch of matrices
    x = np.random.randn(4, 3, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 4D int64 stack of matrices
    x = np.arange(24, dtype=np.int64).reshape((2, 3, 2, 2))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D complex64 matrix
    x = np.array([[1 + 2j, 3 - 4j], [-5j, 6 + 0j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D boolean matrix
    x = np.array([[True, False, True], [False, False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 5D float32 tensor
    x = np.random.uniform(-10.0, 10.0, size=(2, 1, 3, 4, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D row matrix (1, N)
    x = np.array([[10, 20, 30, 40]], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D column matrix (M, 1)
    x = np.array([[-1.5], [2.5], [-3.5]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 3D complex128 stack of matrices
    x = (np.random.randn(3, 2, 3) + 1j * np.random.randn(3, 2, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D with empty dimensions
    x = np.zeros((2, 0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.matrix_transpose"] = matrix_transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matrix_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matrix_transpose'.")


check_valid('jax.numpy.linalg.matrix_transpose', generated_inputs['jax.numpy.linalg.matrix_transpose'], lib="jax", suffix=0)
