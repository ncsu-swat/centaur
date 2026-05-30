
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iterable_inputs():
    list_of_inputs = []

    # Input 1: Simple alphabetic string
    input_dict = {"y": "hello"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty string
    input_dict = {"y": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single character string
    input_dict = {"y": "a"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Numeric string
    input_dict = {"y": "12345"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String representing a library name
    input_dict = {"y": "numpy"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Short string
    input_dict = {"y": "jax"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: String representation of function name
    input_dict = {"y": "iterable"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Long alphabetical string
    input_dict = {"y": "abcdefghijklmnopqrstuvwxyz"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String with special characters and spaces
    input_dict = {"y": "hello world! 123"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String with underscores
    input_dict = {"y": "test_input_string"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.iterable_7"] = iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_7'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_7'], lib="jax", suffix=7)
