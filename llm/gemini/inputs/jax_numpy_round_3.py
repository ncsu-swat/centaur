
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def round_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {"a": 0, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {"a": 5, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {"a": -5, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {"a": 42, "decimals": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {"a": -100, "decimals": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {"a": 12345, "decimals": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {"a": -9876, "decimals": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {"a": 1, "decimals": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {"a": 2147483647, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {"a": -2147483648, "decimals": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_dict = {"a": 999999, "decimals": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.round_3"] = round_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.round_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.round_3'.")


check_valid('jax.numpy.round', generated_inputs['jax.numpy.round_3'], lib="jax", suffix=3)
