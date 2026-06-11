
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def digamma_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar-like)
    x = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another 0D array
    x = np.array(2.5, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array (2x2)
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array (larger)
    x = np.abs(np.random.randn(5, 5)).astype(np.float32) + 0.5
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array
    x = np.abs(np.random.randn(2, 3, 4)).astype(np.float32) + 0.5
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array
    x = np.abs(np.random.randn(2, 2, 2, 2)).astype(np.float32) + 0.5
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 1D array
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 2D array
    x = np.abs(np.random.randn(3, 3)).astype(np.float64) + 0.5
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array
    x = np.abs(np.random.randn(2, 1, 3, 1, 2)).astype(np.float32) + 0.5
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array with positive values
    x = np.array([0.1, 0.5, 1.5, 10.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.digamma_4"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.digamma_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.digamma_4'.")


check_valid('jax.lax.digamma', generated_inputs['jax.lax.digamma_4'], lib="jax", suffix=4)
