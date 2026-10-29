
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hessenberg_inputs():
    list_of_inputs = []

    # Input 1: Basic 4x4 float32 matrix
    a = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 4.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 4.0],
            [2.0, 3.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    input_dict = {
        "a": a,
        "calc_q": False,
        "overwrite_a": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4x4 float64 with calc_q=True
    a = np.random.randn(4, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1x1 edge case matrix
    a = np.array([[5.5]], dtype=np.float32)
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": True,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2x2 matrix with negative numbers
    a = np.array([[-3.0, 4.0], [-1.5, -2.5]], dtype=np.float32)
    input_dict = {
        "a": a,
        "calc_q": False,
        "overwrite_a": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64 3x3 matrix
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex128 4x4 matrix
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(
        np.complex128
    )
    input_dict = {
        "a": a,
        "calc_q": False,
        "overwrite_a": True,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batched 3D float32 tensor
    a = np.random.randn(3, 5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batched 4D float64 tensor
    a = np.random.randn(2, 3, 4, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "calc_q": False,
        "overwrite_a": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 8x8 float32 matrix
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": True,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batched complex64 tensor
    a = (np.random.randn(2, 4, 4) + 1j * np.random.randn(2, 4, 4)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 6x6 float64 matrix with overwrite_a and calc_q True
    a = np.random.randn(6, 6).astype(np.float64)
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": True,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.hessenberg"] = hessenberg_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.hessenberg' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.hessenberg'.")


check_valid('jax.scipy.linalg.hessenberg', generated_inputs['jax.scipy.linalg.hessenberg'], lib="jax", suffix=0)
