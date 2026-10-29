
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def solve_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 system with 1D RHS, generic matrix
    rng = np.random.RandomState(0)
    a = (rng.randn(3, 3) + 5 * np.eye(3)).astype(np.float32)
    b = rng.randn(3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "lower": False,
        "overwrite_a": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
        "assume_a": "gen",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 system with 2D RHS (multiple RHS vectors)
    a = (rng.randn(4, 4) + 6 * np.eye(4)).astype(np.float64)
    b = rng.randn(4, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "lower": True,
        "overwrite_a": True,
        "overwrite_b": False,
        "debug": False,
        "check_finite": False,
        "assume_a": "gen",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batched 3D float32 matrices with batched 2D RHS
    a_raw = rng.randn(2, 3, 3).astype(np.float32)
    a = a_raw + 5 * np.eye(3, dtype=np.float32)
    b = rng.randn(2, 3, 1).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "lower": False,
        "overwrite_a": False,
        "overwrite_b": True,
        "debug": False,
        "check_finite": True,
        "assume_a": "gen",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive definite matrix, assume_a='pos', lower=False
    temp = rng.randn(5, 5).astype(np.float32)
    a = (temp @ temp.T + 2 * np.eye(5)).astype(np.float32)
    b = rng.randn(5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "lower": False,
        "overwrite_a": False,
        "overwrite_b": False,
        "debug": True,
        "check_finite": True,
        "assume_a": "pos",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Positive definite matrix, assume_a='pos', lower=True, 2D RHS
    temp = rng.randn(4, 4).astype(np.float64)
    a = (temp @ temp.T + 3 * np.eye(4)).astype(np.float64)
    b = rng.randn(4, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "lower": True,
        "overwrite_a": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": False,
        "assume_a": "pos",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Symmetric matrix, assume_a='sym', lower=False
    temp = rng.randn(4, 4).astype(np.float64)
    a = (temp + temp.T + 5 * np.eye(4)).astype(np.float64)
    b = rng.randn(4).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "lower": False,
        "overwrite_a": True,
        "overwrite_b": True,
        "debug": False,
        "check_finite": True,
        "assume_a": "sym",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batched symmetric matrix, assume_a='sym', lower=True
    temp = rng.randn(3, 3, 3).astype(np.float32)
    a = temp + np.swapaxes(temp, -1, -2) + 6 * np.eye(3, dtype=np.float32)
    b = rng.randn(3, 3, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "lower": True,
        "overwrite_a": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
        "assume_a": "sym",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex Hermitian matrix, assume_a='her', lower=False
    temp = rng.randn(3, 3) + 1j * rng.randn(3, 3)
    a = (temp + temp.conj().T + 6 * np.eye(3)).astype(np.complex64)
    b = (rng.randn(3) + 1j * rng.randn(3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "lower": False,
        "overwrite_a": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
        "assume_a": "her",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex generic matrix, assume_a='gen', complex128
    temp = rng.randn(4, 4) + 1j * rng.randn(4, 4)
    a = (temp + 5 * np.eye(4)).astype(np.complex128)
    b = (rng.randn(4, 2) + 1j * rng.randn(4, 2)).astype(np.complex128)
    input_dict = {
        "a": a,
        "b": b,
        "lower": False,
        "overwrite_a": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": False,
        "assume_a": "gen",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D batched system with negative values, float32
    a_raw = rng.randn(2, 2, 3, 3).astype(np.float32) - 1.0
    a = a_raw + 8 * np.eye(3, dtype=np.float32)
    b = rng.randn(2, 2, 3, 1).astype(np.float32) - 2.0
    input_dict = {
        "a": a,
        "b": b,
        "lower": False,
        "overwrite_a": False,
        "overwrite_b": False,
        "debug": False,
        "check_finite": True,
        "assume_a": "gen",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.solve"] = solve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.solve'.")


check_valid('jax.scipy.linalg.solve', generated_inputs['jax.scipy.linalg.solve'], lib="jax", suffix=0)
