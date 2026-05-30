
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sinh_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive integer
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small negative integer
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Medium positive integer
    input_dict = {"x": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Medium negative integer
    input_dict = {"x": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger positive integer
    input_dict = {"x": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger negative integer
    input_dict = {"x": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another positive integer
    input_dict = {"x": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another negative integer
    input_dict = {"x": -3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger integer value
    input_dict = {"x": 15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.sinh_3"] = sinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sinh_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sinh_3'.")


check_valid('jax.numpy.sinh', generated_inputs['jax.numpy.sinh_3'], lib="jax", suffix=3)
