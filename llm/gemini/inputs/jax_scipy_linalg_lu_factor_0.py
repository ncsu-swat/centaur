
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lu_factor_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D square float32 matrix
    a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D square float64 matrix with overwrite_a=True
    a = np.array(
        [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
        dtype=np.float64,
    )
    input_dict = {"a": a, "overwrite_a": True, "check_finite": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tall rectangular matrix (M > N)
    a = np.random.randn(5, 3).astype(np.float32)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Wide rectangular matrix (M < N)
    a = np.random.randn(3, 6).astype(np.float32)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D batched square float32 matrices
    a = np.random.randn(4, 3, 3).astype(np.float32)
    input_dict = {"a": a, "overwrite_a": True, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D batched rectangular float64 matrices
    a = np.random.randn(2, 4, 3).astype(np.float64)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D batched square float32 matrices
    a = np.random.randn(2, 3, 4, 4).astype(np.float32)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64 square matrix
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    input_dict = {"a": a, "overwrite_a": True, "check_finite": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex128 rectangular matrix
    a = (np.random.randn(5, 3) + 1j * np.random.randn(5, 3)).astype(
        np.complex128
    )
    input_dict = {"a": a, "overwrite_a": False, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1x1 matrix edge case
    a = np.array([[-5.5]], dtype=np.float32)
    input_dict = {"a": a, "overwrite_a": True, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D batched complex64 matrix
    a = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(
        np.complex64
    )
    input_dict = {"a": a, "overwrite_a": False, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.lu_factor"] = lu_factor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.lu_factor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.lu_factor'.")


check_valid('jax.scipy.linalg.lu_factor', generated_inputs['jax.scipy.linalg.lu_factor'], lib="jax", suffix=0)
