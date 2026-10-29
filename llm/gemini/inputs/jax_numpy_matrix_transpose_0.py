
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def matrix_transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix (float32)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D rectangular matrix with negative values (int32)
    x = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 3D batched matrices (float64)
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 4D batched matrices (float16)
    x = np.random.randn(2, 5, 3, 3).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: Complex numbers 2D matrix
    x = np.array([[1 + 2j, 3 - 4j], [5 + 6j, -7j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 5D high-dimensional tensor
    x = np.ones((2, 2, 2, 4, 6), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Boolean matrix
    x = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Matrix with dimensions of size 1
    x = np.zeros((1, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty matrix with 0 in shape
    x = np.empty((3, 0, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 3D array with int8 values
    x = np.arange(24, dtype=np.int8).reshape((2, 4, 3))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Complex128 batch matrix
    x = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.matrix_transpose"] = matrix_transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.matrix_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.matrix_transpose'.")


check_valid('jax.numpy.matrix_transpose', generated_inputs['jax.numpy.matrix_transpose'], lib="jax", suffix=0)
