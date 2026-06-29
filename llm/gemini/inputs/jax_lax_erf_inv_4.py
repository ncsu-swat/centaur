
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def erf_inv_inputs():
    list_of_inputs = []

    # Input 1: Scalar representation of True (1.0) as float32
    x = np.array(True, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: Scalar representation of False (0.0) as float32
    x = np.array(False, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D array from booleans to float32
    x = np.array([True, False, True, False], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D array from booleans to float32
    x = np.array([[True, False], [False, True]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array from booleans to float64
    x = np.array([[[True, True], [False, False]], [[False, True], [True, False]]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D array of ones (True) to float32
    x = np.ones((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 5D array of zeros (False) to float32
    x = np.zeros((1, 2, 1, 3, 2), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Empty 1D array (float32)
    x = np.array([], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array with True (1.0) float64
    x = np.array([True, True, True, True, True], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D array with False (0.0) float64
    x = np.array([False, False, False], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 2D array from random booleans to float32
    x = np.random.choice([True, False], size=(5, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.erf_inv_4"] = erf_inv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erf_inv_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erf_inv_4'.")


check_valid('jax.lax.erf_inv', generated_inputs['jax.lax.erf_inv_4'], lib="jax", suffix=4)
