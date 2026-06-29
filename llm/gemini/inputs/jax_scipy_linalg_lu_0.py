
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lu_inputs():
    list_of_inputs = []

    # Input 1: Square matrix, float32, default boolean values
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "permute_l": False,
        "overwrite_a": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tall rectangular matrix, float32, permute_l=True
    a = np.random.randn(6, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "permute_l": True,
        "overwrite_a": False,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Wide rectangular matrix, float64, overwrite_a=True
    a = np.random.randn(3, 7).astype(np.float64)
    input_dict = {
        "a": a,
        "permute_l": False,
        "overwrite_a": True,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Square matrix, float32, permute_l=True, overwrite_a=True, check_finite=False
    a = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "permute_l": True,
        "overwrite_a": True,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Square matrix, float64, check_finite=True
    a = np.random.randn(8, 8).astype(np.float64)
    input_dict = {
        "a": a,
        "permute_l": False,
        "overwrite_a": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small 2x2 matrix, float32, all parameters True
    a = np.array([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "permute_l": True,
        "overwrite_a": True,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large matrix, float32
    a = np.random.randn(50, 50).astype(np.float32)
    input_dict = {
        "a": a,
        "permute_l": False,
        "overwrite_a": False,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tall rectangular matrix, float64
    a = np.random.randn(8, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "permute_l": True,
        "overwrite_a": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Wide rectangular matrix, float32
    a = np.random.randn(4, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "permute_l": False,
        "overwrite_a": True,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Matrix with negative and zero values, float64
    a = np.array([[-1.0, 0.0, 3.0], [4.0, -5.0, 0.0], [0.0, 7.0, -9.0]], dtype=np.float64)
    input_dict = {
        "a": a,
        "permute_l": True,
        "overwrite_a": True,
        "check_finite": True
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
