
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arcsin_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive boundary value
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative boundary value
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive integer outside [-1, 1]
    input_dict = {"x": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small negative integer outside [-1, 1]
    input_dict = {"x": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Moderate positive integer
    input_dict = {"x": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Moderate negative integer
    input_dict = {"x": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger positive integer
    input_dict = {"x": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger negative integer
    input_dict = {"x": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High magnitude positive integer
    input_dict = {"x": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: High magnitude negative integer
    input_dict = {"x": -100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.arcsin_3"] = arcsin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arcsin_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arcsin_3'.")


check_valid('jax.numpy.arcsin', generated_inputs['jax.numpy.arcsin_3'], lib="jax", suffix=3)
