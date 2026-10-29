
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lu_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D square matrix (float32)
    a = np.array([[1.0, 2.0, 3.0], [5.0, 4.0, 2.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "permute_l": False,
        "overwrite_a": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tall rectangular matrix (float64) with permute_l=True
    a = np.random.randn(5, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "permute_l": True,
        "overwrite_a": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Wide rectangular matrix (float32)
    a = np.random.randn(3, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "permute_l": False,
        "overwrite_a": True,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2x2 matrix with negative numbers (float64)
    a = np.array([[-4.5, 2.3], [1.1, -9.8]], dtype=np.float64)
    input_dict = {
        "a": a,
        "permute_l": False,
        "overwrite_a": True,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1x1 matrix (float32)
    a = np.array([[7.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "permute_l": True,
        "overwrite_a": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single row matrix (1x5)
    a = np.random.randn(1, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "permute_l": False,
        "overwrite_a": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single column matrix (5x1)
    a = np.random.randn(5, 1).astype(np.float64)
    input_dict = {
        "a": a,
        "permute_l": True,
        "overwrite_a": True,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64 square matrix
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    input_dict = {
        "a": a,
        "permute_l": False,
        "overwrite_a": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex128 rectangular matrix (permute_l=True)
    a = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(np.complex128)
    input_dict = {
        "a": a,
        "permute_l": True,
        "overwrite_a": False,
        "check_finite": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger square matrix (8x8 float32)
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "permute_l": False,
        "overwrite_a": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Rectangular matrix with positive and negative floats
    a = (np.random.randn(6, 4) * 10).astype(np.float64)
    input_dict = {
        "a": a,
        "permute_l": True,
        "overwrite_a": False,
        "check_finite": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.lu"] = lu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.lu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.lu'.")


check_valid('jax.scipy.linalg.lu', generated_inputs['jax.scipy.linalg.lu'], lib="jax", suffix=0)
