
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hessenberg_inputs():
    list_of_inputs = []

    # Input 1: Basic 3x3 float32 matrix with calc_q=True
    a = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4x4 float64 matrix with calc_q=False and overwrite_a=True
    a = np.random.randn(4, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "calc_q": False,
        "overwrite_a": True,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 5x5 complex64 matrix with calc_q=True
    a_real = np.random.randn(5, 5).astype(np.float32)
    a_imag = np.random.randn(5, 5).astype(np.float32)
    a = a_real + 1j * a_imag
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched 3x3 float32 matrices (shape 2, 3, 3)
    a = np.random.randn(2, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "calc_q": False,
        "overwrite_a": False,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: High dimensional batch (3, 2, 4, 4) float64
    a = np.random.randn(3, 2, 4, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": True,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2x2 complex128 matrix
    a_real = np.random.randn(2, 2).astype(np.float64)
    a_imag = np.random.randn(2, 2).astype(np.float64)
    a = a_real + 1j * a_imag
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": False,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 8x8 float32 matrix with larger values
    a = (np.random.randn(8, 8) * 10.0).astype(np.float32)
    input_dict = {
        "a": a,
        "calc_q": False,
        "overwrite_a": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Edge case - 1x1 float32 matrix
    a = np.random.randn(1, 1).astype(np.float32)
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large batch of 5x5 complex64 matrices (shape 10, 5, 5)
    a_real = np.random.randn(10, 5, 5).astype(np.float32)
    a_imag = np.random.randn(10, 5, 5).astype(np.float32)
    a = a_real + 1j * a_imag
    input_dict = {
        "a": a,
        "calc_q": False,
        "overwrite_a": True,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 6x6 float64 matrix with a mix of positive and negative elements
    a = np.random.uniform(-5.0, 5.0, (6, 6)).astype(np.float64)
    input_dict = {
        "a": a,
        "calc_q": True,
        "overwrite_a": False,
        "check_finite": False
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
