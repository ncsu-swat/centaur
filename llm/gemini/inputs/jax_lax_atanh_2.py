
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atanh_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive float
    input_dict = {"x": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float
    input_dict = {"x": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Close to 1
    input_dict = {"x": 0.99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Close to -1
    input_dict = {"x": -0.99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small positive float
    input_dict = {"x": 1e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small negative float
    input_dict = {"x": -1e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another positive float
    input_dict = {"x": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another negative float
    input_dict = {"x": -0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High precision float near 1
    input_dict = {"x": 0.9999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: High precision float near -1
    input_dict = {"x": -0.9999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.atanh_2"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.atanh_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.atanh_2'.")


check_valid('jax.lax.atanh', generated_inputs['jax.lax.atanh_2'], lib="jax", suffix=2)
