
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def erf_inv_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive mid-range value
    input_dict = {"x": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative mid-range value
    input_dict = {"x": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive value
    input_dict = {"x": 0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small negative value
    input_dict = {"x": -0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive value close to 1
    input_dict = {"x": 0.9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large negative value close to -1
    input_dict = {"x": -0.9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Value very close to 1
    input_dict = {"x": 0.99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Value very close to -1
    input_dict = {"x": -0.99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Standard float value
    input_dict = {"x": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Standard negative float value
    input_dict = {"x": -0.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.erf_inv_2"] = erf_inv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erf_inv_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erf_inv_2'.")


check_valid('jax.lax.erf_inv', generated_inputs['jax.lax.erf_inv_2'], lib="jax", suffix=2)
