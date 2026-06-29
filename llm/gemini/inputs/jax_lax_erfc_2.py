
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def erfc_inputs():
    list_of_inputs = []

    # Input 1: Scalar 0.0 (float32)
    input_dict = {"x": np.array(0.0, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar negative integer value (float64)
    input_dict = {"x": np.array(-5.0, dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of positive integer values (float32)
    input_dict = {"x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of mixed integer values (float32)
    input_dict = {"x": np.array([-10.0, 0.0, 10.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array of integer values (float32)
    input_dict = {"x": np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array with larger integer values (float64)
    input_dict = {"x": np.array([[-100.0, 100.0], [0.0, 50.0]], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array (float32)
    input_dict = {"x": np.array([[[1.0, -1.0], [2.0, -2.0]]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array (float32)
    input_dict = {"x": np.array([[[[-5.0], [5.0]]]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive and negative integer values (float64)
    input_dict = {"x": np.array([1000.0, -1000.0], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with zero (float32)
    input_dict = {"x": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.erfc_2"] = erfc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erfc_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erfc_2'.")


check_valid('jax.lax.erfc', generated_inputs['jax.lax.erfc_2'], lib="jax", suffix=2)
