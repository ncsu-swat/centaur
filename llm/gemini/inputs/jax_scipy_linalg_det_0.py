
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def det_inputs():
    list_of_inputs = []

    # Input 1
    a = np.random.randn(2, 2).astype(np.float32)
    list_of_inputs.append({"a": a, "overwrite_a": False, "check_finite": True})

    # Input 2
    a = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({"a": a, "overwrite_a": True, "check_finite": False})

    # Input 3
    a = np.random.randn(2, 2).astype(np.float64)
    list_of_inputs.append({"a": a, "overwrite_a": False, "check_finite": False})

    # Input 4
    a = np.random.randn(3, 3).astype(np.float64)
    list_of_inputs.append({"a": a, "overwrite_a": True, "check_finite": True})

    # Input 5
    a = np.random.randn(2, 2, 2).astype(np.float32)
    list_of_inputs.append({"a": a, "overwrite_a": False, "check_finite": True})

    # Input 6
    a = np.random.randn(2, 3, 3).astype(np.float32)
    list_of_inputs.append({"a": a, "overwrite_a": True, "check_finite": False})

    # Input 7
    a = np.random.randn(2, 2, 2, 2).astype(np.float64)
    list_of_inputs.append({"a": a, "overwrite_a": False, "check_finite": True})

    # Input 8
    a = np.eye(2).astype(np.float32)
    list_of_inputs.append({"a": a, "overwrite_a": False, "check_finite": False})

    # Input 9
    a = np.eye(3).astype(np.float64)
    list_of_inputs.append({"a": a, "overwrite_a": True, "check_finite": True})

    # Input 10
    a = np.random.randn(1, 1).astype(np.float32)
    list_of_inputs.append({"a": a, "overwrite_a": False, "check_finite": True})

    return list_of_inputs

generated_inputs["jax.scipy.linalg.det"] = det_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.det' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.det'.")


check_valid('jax.scipy.linalg.det', generated_inputs['jax.scipy.linalg.det'], lib="jax", suffix=0)
