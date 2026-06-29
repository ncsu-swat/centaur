
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def qr_multiply_inputs():
    list_of_inputs = []

    # Case 1
    a = np.random.randn(4, 3).astype(np.float32)
    c = np.random.randn(4).astype(np.float32)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "right",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2
    a = np.random.randn(4, 3).astype(np.float32)
    c = np.random.randn(3).astype(np.float32)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "left",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3
    a = np.random.randn(5, 5).astype(np.float64)
    c = np.random.randn(2, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "right",
        "pivoting": True,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4
    a = np.random.randn(5, 5).astype(np.float64)
    c = np.random.randn(5, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "left",
        "pivoting": True,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5
    a = np.random.randn(2, 4, 3).astype(np.float32)
    c = np.random.randn(2, 2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "right",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": True,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6
    a = np.random.randn(2, 3, 4).astype(np.float32)
    c = np.random.randn(2, 3, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "left",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    c = (np.random.randn(4, 2) + 1j * np.random.randn(4, 2)).astype(np.complex64)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "left",
        "pivoting": False,
        "conjugate": True,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8
    a = np.random.randn(6, 2).astype(np.float64)
    c = np.random.randn(6).astype(np.float64)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "right",
        "pivoting": True,
        "conjugate": False,
        "overwrite_a": True,
        "overwrite_c": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9
    a = np.random.randn(3, 6).astype(np.float32)
    c = np.random.randn(3, 1).astype(np.float32)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "left",
        "pivoting": False,
        "conjugate": False,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10
    a = (np.random.randn(5, 3) + 1j * np.random.randn(5, 3)).astype(np.complex128)
    c = (np.random.randn(2, 5) + 1j * np.random.randn(2, 5)).astype(np.complex128)
    input_dict = {
        "a": a,
        "c": c,
        "mode": "right",
        "pivoting": True,
        "conjugate": True,
        "overwrite_a": False,
        "overwrite_c": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.qr_multiply"] = qr_multiply_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.qr_multiply' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.qr_multiply'.")


check_valid('jax.scipy.linalg.qr_multiply', generated_inputs['jax.scipy.linalg.qr_multiply'], lib="jax", suffix=0)
