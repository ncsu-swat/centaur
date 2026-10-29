
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isscalar_inputs():
    list_of_inputs = []

    # Input 1: Empty tuple
    input_dict = {"element": ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1-tuple of integer
    input_dict = {"element": (1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple of negative and positive integers
    input_dict = {"element": (-42, 0, 100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tuple of floats
    input_dict = {"element": (1.5, -3.14, 0.0, 2.718)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple of single negative float
    input_dict = {"element": (-10.5,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tuple of zeros
    input_dict = {"element": (0, 0, 0, 0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tuple of strictly negative integers
    input_dict = {"element": (-100, -50, -25, -1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tuple of positive floats
    input_dict = {"element": (0.1, 0.2, 0.3, 0.4, 0.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tuple of mixed signs float
    input_dict = {"element": (-1.0, 1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Two-element tuple of integers
    input_dict = {"element": (42, -42)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Tuple of large integers
    input_dict = {"element": (1000, 2000, 3000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isscalar_7"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_7'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_7'], lib="jax", suffix=7)
