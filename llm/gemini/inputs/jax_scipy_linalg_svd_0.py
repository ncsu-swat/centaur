
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def svd_inputs():
    list_of_inputs = []

    # Input 1: 2D square float32 matrix, default parameters
    a = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    input_dict = {
        "a": a,
        "full_matrices": True,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D rectangular (tall) float64 matrix with full_matrices=False
    a = np.random.randn(5, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": False,
        "lapack_driver": "gesvd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D rectangular (wide) float32 matrix with compute_uv=False
    a = np.random.randn(2, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "full_matrices": True,
        "compute_uv": False,
        "overwrite_a": True,
        "check_finite": True,
        "lapack_driver": "gesdd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex64 2D matrix
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex128 2D matrix with negative values
    a = (
        -np.random.uniform(1.0, 10.0, size=(3, 5))
        + 1j * np.random.randn(3, 5)
    ).astype(np.complex128)
    input_dict = {
        "a": a,
        "full_matrices": True,
        "compute_uv": False,
        "overwrite_a": False,
        "check_finite": False,
        "lapack_driver": "gesvd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D batched float32 array
    a = np.random.randn(3, 4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "full_matrices": True,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D batched float64 array with full_matrices=False
    a = np.random.randn(2, 2, 5, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "overwrite_a": True,
        "check_finite": True,
        "lapack_driver": "gesdd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batched complex64 array with compute_uv=False
    a = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": False,
        "overwrite_a": False,
        "check_finite": False,
        "lapack_driver": "gesvd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2x2 small matrix with negative float32 values
    a = np.array([[-3.5, -1.2], [-0.5, -4.8]], dtype=np.float32)
    input_dict = {
        "a": a,
        "full_matrices": True,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D batched float32 tensor
    a = np.random.randn(1, 2, 2, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "overwrite_a": False,
        "check_finite": True,
        "lapack_driver": "gesdd",
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
