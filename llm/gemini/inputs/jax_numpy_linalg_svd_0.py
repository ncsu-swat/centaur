
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def svd_inputs():
    list_of_inputs = []

    # Input 1: 2D (0, 0) float32 matrix, compute_uv=True
    a = np.empty((0, 0), dtype=np.float32)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D (0, 4) float32 matrix, compute_uv=True
    a = np.empty((0, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D (5, 0) float64 matrix, compute_uv=False
    a = np.empty((5, 0), dtype=np.float64)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": False,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D (0, 1) float64 matrix, compute_uv=True
    a = np.empty((0, 1), dtype=np.float64)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched 3D (2, 0, 0) float32 array, compute_uv=True
    a = np.empty((2, 0, 0), dtype=np.float32)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched 3D (3, 0, 4) float32 array, compute_uv=False
    a = np.empty((3, 0, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": False,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batched 3D (2, 5, 0) float64 array, compute_uv=True
    a = np.empty((2, 5, 0), dtype=np.float64)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D (2, 3, 0, 0) float32 array, compute_uv=True
    a = np.empty((2, 3, 0, 0), dtype=np.float32)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D (1, 2, 0, 3) complex64 array, compute_uv=True
    a = np.empty((1, 2, 0, 3), dtype=np.complex64)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D (2, 2, 4, 0) complex128 array, compute_uv=False
    a = np.empty((2, 2, 4, 0), dtype=np.complex128)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": False,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D (1, 0) float32 matrix, compute_uv=True
    a = np.empty((1, 0), dtype=np.float32)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": True,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 2D (0, 0) complex64 matrix, compute_uv=False
    a = np.empty((0, 0), dtype=np.complex64)
    input_dict = {
        "a": a,
        "full_matrices": False,
        "compute_uv": False,
        "hermitian": False,
        "subset_by_index": (0, 0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.svd"] = svd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.svd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.svd'.")


check_valid('jax.numpy.linalg.svd', generated_inputs['jax.numpy.linalg.svd'], lib="jax", suffix=0)
