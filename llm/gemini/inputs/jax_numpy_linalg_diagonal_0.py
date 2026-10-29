
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_numpy_linalg_diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix, float32, offset 0
    x = np.random.randn(4, 4).astype(np.float32)
    offset = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "offset": offset}))

    # Input 2: 2D rectangular matrix (wide), int32, positive offset
    x = np.arange(12, dtype=np.int32).reshape(3, 4)
    offset = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "offset": offset}))

    # Input 3: 2D rectangular matrix (tall), float64, negative offset
    x = np.random.randn(5, 3).astype(np.float64)
    offset = -2
    list_of_inputs.append(copy.deepcopy({"x": x, "offset": offset}))

    # Input 4: 3D batched square matrices, float32, offset 0
    x = np.random.randn(2, 6, 6).astype(np.float32)
    offset = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "offset": offset}))

    # Input 5: 3D batched rectangular matrices, int64, positive offset
    x = np.arange(60, dtype=np.int64).reshape(3, 4, 5)
    offset = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "offset": offset}))

    # Input 6: 4D batched matrices, float32, negative offset
    x = np.random.randn(2, 3, 5, 5).astype(np.float32)
    offset = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "offset": offset}))

    # Input 7: 2D boolean matrix, offset 0
    x = np.random.choice([True, False], size=(4, 4))
    offset = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "offset": offset}))

    # Input 8: Complex matrix, complex64, offset 1
    x = (
        np.random.randn(4, 4).astype(np.float32)
        + 1j * np.random.randn(4, 4).astype(np.float32)
    ).astype(np.complex64)
    offset = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "offset": offset}))

    # Input 9: Large 2D matrix, float32, large offset
    x = np.random.randn(20, 20).astype(np.float32)
    offset = 10
    list_of_inputs.append(copy.deepcopy({"x": x, "offset": offset}))

    # Input 10: 5D batched matrix, float64, negative offset
    x = np.random.randn(2, 2, 2, 3, 3).astype(np.float64)
    offset = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "offset": offset}))

    # Input 11: 2D matrix, offset equals row size minus 1
    x = np.random.randn(3, 3).astype(np.float32)
    offset = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "offset": offset}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.diagonal"] = (
    jax_numpy_linalg_diagonal_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.diagonal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.diagonal'.")


check_valid('jax.numpy.linalg.diagonal', generated_inputs['jax.numpy.linalg.diagonal'], lib="jax", suffix=0)
