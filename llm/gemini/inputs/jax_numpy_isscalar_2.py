
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isscalar_inputs():
    list_of_inputs = []

    # Input 1: zero integer
    input_dict = {"element": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive integer
    input_dict = {"element": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative integer
    input_dict = {"element": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: small positive integer
    input_dict = {"element": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: small negative integer
    input_dict = {"element": -100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: large positive integer
    input_dict = {"element": 1000000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large negative integer
    input_dict = {"element": -1000000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: powers of two
    input_dict = {"element": 256}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: negative power of two
    input_dict = {"element": -512}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 16-bit max integer
    input_dict = {"element": 32767}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 32-bit max integer
    input_dict = {"element": 2147483647}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 32-bit min integer
    input_dict = {"element": -2147483648}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isscalar_2"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_2'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_2'], lib="jax", suffix=2)
