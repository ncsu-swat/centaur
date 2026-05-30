
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arctanh_inputs():
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

    # Input 4: Float close to 1
    input_dict = {"x": 0.99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float close to -1
    input_dict = {"x": -0.99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small positive float
    input_dict = {"x": 0.12345}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small negative float
    input_dict = {"x": -0.12345}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small float
    input_dict = {"x": 1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another positive float
    input_dict = {"x": 0.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another negative float
    input_dict = {"x": -0.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Float near boundary
    input_dict = {"x": 0.999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.arctanh_2"] = arctanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arctanh_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arctanh_2'.")


check_valid('jax.numpy.arctanh', generated_inputs['jax.numpy.arctanh_2'], lib="jax", suffix=2)
