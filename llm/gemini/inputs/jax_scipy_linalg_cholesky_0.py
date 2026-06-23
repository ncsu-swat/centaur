
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cholesky_inputs():
    list_of_inputs = []

    # Input 1
    a = (np.eye(2) * 1.0).astype(np.float32)
    input_dict = {
        "a": a,
        "lower": True,
        "overwrite_a": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = (np.eye(2) * 2.0).astype(np.float32)
    input_dict = {
        "a": a,
        "lower": False,
        "overwrite_a": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = (np.eye(2) * 3.0).astype(np.float32)
    input_dict = {
        "a": a,
        "lower": True,
        "overwrite_a": True,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = (np.eye(2) * 4.0).astype(np.float32)
    input_dict = {
        "a": a,
        "lower": False,
        "overwrite_a": True,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = (np.eye(2) * 5.0).astype(np.float32)
    input_dict = {
        "a": a,
        "lower": True,
        "overwrite_a": False,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = (np.eye(2) * 6.0).astype(np.float32)
    input_dict = {
        "a": a,
        "lower": False,
        "overwrite_a": False,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = (np.eye(2) * 7.0).astype(np.float32)
    input_dict = {
        "a": a,
        "lower": True,
        "overwrite_a": True,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = (np.eye(2) * 8.0).astype(np.float32)
    input_dict = {
        "a": a,
        "lower": False,
        "overwrite_a": True,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = (np.eye(2) * 9.0).astype(np.float32)
    input_dict = {
        "a": a,
        "lower": True,
        "overwrite_a": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = (np.eye(2) * 10.0).astype(np.float32)
    input_dict = {
        "a": a,
        "lower": False,
        "overwrite_a": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.cholesky"] = cholesky_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.cholesky' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.cholesky'.")


check_valid('jax.scipy.linalg.cholesky', generated_inputs['jax.scipy.linalg.cholesky'], lib="jax", suffix=0)
