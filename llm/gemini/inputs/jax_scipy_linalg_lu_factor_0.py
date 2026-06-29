
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lu_factor_inputs():
    list_of_inputs = []

    # Input 1: Small float32 square matrix
    a = np.random.randn(2, 2).astype(np.float32)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64 3x3 square matrix with overwrite_a=True
    a = np.random.randn(3, 3).astype(np.float64)
    input_dict = {"a": a, "overwrite_a": True, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float32 5x5 matrix with check_finite=False, containing negative values
    a = (np.random.randn(5, 5) * 10).astype(np.float32)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64 square matrix
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex128 square matrix
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex128)
    input_dict = {"a": a, "overwrite_a": True, "check_finite": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched 3D matrix (batch size 2, 4x4 matrix)
    a = np.random.randn(2, 4, 4).astype(np.float32)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rectangular matrix 3x4 (M < N)
    a = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Rectangular matrix 4x3 (M > N)
    a = np.random.randn(4, 3).astype(np.float64)
    input_dict = {"a": a, "overwrite_a": True, "check_finite": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 64x64 matrix
    a = np.random.randn(64, 64).astype(np.float32)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional batched matrix (5, 2, 3, 3)
    a = np.random.randn(5, 2, 3, 3).astype(np.float32)
    input_dict = {"a": a, "overwrite_a": False, "check_finite": False}
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
