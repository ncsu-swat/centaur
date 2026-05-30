
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def square_inputs():
    list_of_inputs = []

    # Input 1: Simple positive float
    input_dict = {"x": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative float
    input_dict = {"x": -2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive float
    input_dict = {"x": 1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive float
    input_dict = {"x": 1e5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative float with multiple decimal places
    input_dict = {"x": -123.456}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Pi approximation
    input_dict = {"x": 3.1415926535}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small negative float
    input_dict = {"x": -0.0001}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative zero
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another positive float
    input_dict = {"x": 999.99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.square_3"] = square_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.square_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.square_3'.")


check_valid('jax.numpy.square', generated_inputs['jax.numpy.square_3'], lib="jax", suffix=3)
