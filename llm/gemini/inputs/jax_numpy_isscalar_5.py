
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isscalar_inputs():
    list_of_inputs = []

    # Input 1: Basic alphabetic string
    input_dict = {"element": "hello"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty string
    input_dict = {"element": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Numeric string representing an integer
    input_dict = {"element": "12345"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Numeric string representing a float
    input_dict = {"element": "-3.14159"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean representation string
    input_dict = {"element": "False"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single character string
    input_dict = {"element": "z"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: String with special characters and spaces
    input_dict = {"element": "hello world! @2023"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-line string
    input_dict = {"element": "line one\nline two"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String containing unicode characters
    input_dict = {"element": "π_value_3.14"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String consisting only of whitespace
    input_dict = {"element": "   "}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isscalar_5"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_5'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_5'], lib="jax", suffix=5)
