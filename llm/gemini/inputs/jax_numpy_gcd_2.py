
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def gcd_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integers
    input_dict = {"x1": 12, "x2": 18}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive and negative integers
    input_dict = {"x1": -24, "x2": 36}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero and a positive integer
    input_dict = {"x1": 0, "x2": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive integer and zero
    input_dict = {"x1": 7, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Both zeros
    input_dict = {"x1": 0, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Both negative integers
    input_dict = {"x1": -15, "x2": -25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Coprime integers
    input_dict = {"x1": 101, "x2": 103}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger integers
    input_dict = {"x1": 1000000, "x2": 2500000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: One negative and one positive unit
    input_dict = {"x1": -1, "x2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Equal integers
    input_dict = {"x1": 42, "x2": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Powers of two
    input_dict = {"x1": 128, "x2": 256}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative coprime integers
    input_dict = {"x1": -77, "x2": 121}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.gcd_2"] = gcd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gcd_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gcd_2'.")


check_valid('jax.numpy.gcd', generated_inputs['jax.numpy.gcd_2'], lib="jax", suffix=2)
