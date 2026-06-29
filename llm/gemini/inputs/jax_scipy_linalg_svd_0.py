
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def svd_inputs():
    list_of_inputs = []

    # Input 1: Basic real 2D float32 matrix
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "full_matrices": True,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex 2D matrix
    a = (np.random.randn(4, 6) + 1j * np.random.randn(4, 6)).astype(np.complex64)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64, compute_uv=False
    a = np.random.randn(8, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "full_matrices": True,
        "compute_uv": False,
        "overwrite_a": True,
        "check_finite": False,
        "lapack_driver": "gesvd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D batched matrices
    a = np.random.randn(3, 4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tall matrix (float32)
    a = np.random.randn(10, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "full_matrices": True,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Wide matrix (float64), full_matrices=False
    a = np.random.randn(2, 7).astype(np.float64)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "overwrite_a": True,
        "check_finite": True,
        "lapack_driver": "gesdd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex 128 batch matrix
    a = (np.random.randn(2, 5, 5) + 1j * np.random.randn(2, 5, 5)).astype(np.complex128)
    input_dict = {
        "a": a,
        "full_matrices": True,
        "compute_uv": False,
        "overwrite_a": False,
        "check_finite": False,
        "lapack_driver": "gesvd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D batched matrices
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Square matrix with negative values (float32)
    a = np.random.uniform(-10.0, -1.0, size=(6, 6)).astype(np.float32)
    input_dict = {
        "a": a,
        "full_matrices": True,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Rectangular matrix with small values
    a = np.random.randn(3, 8).astype(np.float32) * 1e-5
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.svd"] = svd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.svd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.svd'.")


check_valid('jax.scipy.linalg.svd', generated_inputs['jax.scipy.linalg.svd'], lib="jax", suffix=0)
