
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def svdvals_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix (float32)
    x = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D rectangular matrix with M < N (float32)
    x = np.random.randn(2, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D rectangular matrix with M > N (float64)
    x = np.random.randn(6, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D batched square matrices (float32)
    x = np.random.randn(4, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D batched rectangular matrices (float64)
    x = np.random.randn(2, 5, 2).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D batched matrices (float32)
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Complex matrix 2D (complex64)
    real_part = np.random.randn(4, 4).astype(np.float32)
    imag_part = np.random.randn(4, 4).astype(np.float32)
    x = (real_part + 1j * imag_part).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Complex batched matrix 3D (complex128)
    real_part = np.random.randn(3, 4, 2).astype(np.float64)
    imag_part = np.random.randn(3, 4, 2).astype(np.float64)
    x = (real_part + 1j * imag_part).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Integer matrix (int32)
    x = np.random.randint(-10, 10, size=(4, 3)).astype(np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Matrix with negative values (float32)
    x = -np.abs(np.random.randn(5, 5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1x1 matrix (float32)
    x = np.array([[3.14]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.svdvals"] = svdvals_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.svdvals' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.svdvals'.")


check_valid('jax.numpy.linalg.svdvals', generated_inputs['jax.numpy.linalg.svdvals'], lib="jax", suffix=0)
