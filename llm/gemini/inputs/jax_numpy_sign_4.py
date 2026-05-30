
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp

# Patch jax.numpy.sign to convert list inputs to jax arrays automatically
original_sign = jnp.sign

def patched_sign(x, *args, **kwargs):
    if isinstance(x, list):
        x = jnp.array(x)
    return original_sign(x, *args, **kwargs)

jnp.sign = patched_sign

def sign_inputs():
    list_of_inputs = []

    # Input 1: Basic list of floats
    input_dict = {"x": [1.0, -2.5, 0.0, 3.14, -0.5]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of integers with negative values
    input_dict = {"x": [-10, 0, 5, -2, 100]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of complex numbers
    input_dict = {"x": [3 + 4j, -1j, 0 + 0j, -5 + 12j]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D nested list
    input_dict = {"x": [[1.0, -1.0, 0.0], [-2.0, 3.0, -4.0]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List with infinity values
    input_dict = {"x": [float('inf'), float('-inf'), 0.0, 1.0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D nested list
    input_dict = {"x": [[[1, -1], [2, -2]], [[3, -3], [4, -4]]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List with zeros and negative zero
    input_dict = {"x": [-1.0, 0.0, 1.0, -0.0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Nested list of complex numbers
    input_dict = {"x": [[1j, -1j], [2 + 2j, -2 - 2j]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List with very small float values
    input_dict = {"x": [1e-15, -1e-15, 0.0, 1.0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D list with mixed numeric types
    input_dict = {"x": [-1, 2.5, -3, 4.2, 0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.sign_4"] = sign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sign_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sign_4'.")


check_valid('jax.numpy.sign', generated_inputs['jax.numpy.sign_4'], lib="jax", suffix=4)
