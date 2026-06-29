
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def qr_inputs():
    list_of_inputs = []

    # Input 1: Square matrix float32, basic defaults
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 0,
        "mode": "full",
        "pivoting": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tall matrix float64, mode "r"
    a = np.random.randn(8, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "overwrite_a": True,
        "lwork": 1,
        "mode": "r",
        "pivoting": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Wide matrix float32, mode "economic", with pivoting
    a = np.random.randn(3, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": -1,
        "mode": "economic",
        "pivoting": True,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D batched matrix float32, mode "full"
    a = np.random.randn(2, 4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 0,
        "mode": "full",
        "pivoting": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64 matrix, mode "economic"
    a = (np.random.randn(6, 6) + 1j * np.random.randn(6, 6)).astype(np.complex64)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 0,
        "mode": "economic",
        "pivoting": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D batched matrix float64
    a = np.random.randn(2, 3, 5, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "overwrite_a": True,
        "lwork": 2,
        "mode": "full",
        "pivoting": False,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Square float32, mode "r" with pivoting
    a = np.random.randn(7, 7).astype(np.float32)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 0,
        "mode": "r",
        "pivoting": True,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex128 3D batched matrix, mode "full" with pivoting
    a = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(np.complex128)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 0,
        "mode": "full",
        "pivoting": True,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small 2D matrix (2x2) float32, mode "economic" with pivoting
    a = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "overwrite_a": True,
        "lwork": 0,
        "mode": "economic",
        "pivoting": True,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tall matrix (10x3) float32, mode "full"
    a = np.random.randn(10, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "overwrite_a": False,
        "lwork": 100,
        "mode": "full",
        "pivoting": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.qr"] = qr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.qr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.qr'.")


check_valid('jax.scipy.linalg.qr', generated_inputs['jax.scipy.linalg.qr'], lib="jax", suffix=0)
