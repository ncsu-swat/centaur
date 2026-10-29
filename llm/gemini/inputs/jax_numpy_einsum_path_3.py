
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def einsum_path_inputs():
    list_of_inputs = []

    # Input 1: 1D identity
    input_dict = {
        "subscripts": "i->i",
        "operands": np.array([-1.5, 2.0, -3.5, 4.0], dtype=np.float32),
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D transpose with negative integers
    input_dict = {
        "subscripts": "ij->ji",
        "operands": (np.arange(6, dtype=np.int32).reshape(2, 3) - 3),
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Extract diagonal from square matrix (float64)
    input_dict = {
        "subscripts": "ii->i",
        "operands": np.random.randn(4, 4).astype(np.float64),
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Trace of matrix (float32)
    input_dict = {
        "subscripts": "ii->",
        "operands": (np.eye(5, dtype=np.float32) * -2.5),
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor axis permutation (complex64)
    input_dict = {
        "subscripts": "ijk->kji",
        "operands": np.ones((2, 3, 4), dtype=np.complex64),
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Partial trace on last two axes of 3D tensor (float32)
    input_dict = {
        "subscripts": "ijj->i",
        "operands": np.random.uniform(-10.0, 10.0, size=(3, 4, 4)).astype(
            np.float32
        ),
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D permutation with int64
    input_dict = {
        "subscripts": "ijkl->ljki",
        "operands": np.arange(24, dtype=np.int64).reshape(2, 3, 2, 2) - 12,
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Sum reduction on 1D vector (int32)
    input_dict = {
        "subscripts": "i->",
        "operands": np.array([-10, 20, -30, 40, -50], dtype=np.int32),
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Full reduction of 2D matrix (float16)
    input_dict = {
        "subscripts": "ij->",
        "operands": np.random.randn(5, 5).astype(np.float16),
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Reduction along middle axis of 3D tensor (complex128)
    input_dict = {
        "subscripts": "ijk->ik",
        "operands": (
            np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)
        ).astype(np.complex128),
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D reverse axes permutation (float32)
    input_dict = {
        "subscripts": "ijklm->mlkji",
        "operands": np.ones((2, 2, 2, 2, 2), dtype=np.float32),
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Partial trace on last two axes of 4D tensor (float64)
    input_dict = {
        "subscripts": "ijkk->ij",
        "operands": np.random.randn(2, 3, 4, 4).astype(np.float64),
        "optimize": [(0,)],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.einsum_path_3"] = einsum_path_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.einsum_path_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.einsum_path_3'.")


check_valid('jax.numpy.einsum_path', generated_inputs['jax.numpy.einsum_path_3'], lib="jax", suffix=3)
