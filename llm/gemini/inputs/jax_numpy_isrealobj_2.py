
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isrealobj_inputs():
    list_of_inputs = []

    # Input 1: zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive small integer
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative small integer
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive two-digit integer
    input_dict = {"x": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative two-digit integer
    input_dict = {"x": -99}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: larger positive integer
    input_dict = {"x": 1024}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: larger negative integer
    input_dict = {"x": -2048}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: max 32-bit signed integer
    input_dict = {"x": 2147483647}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: min 32-bit signed integer
    input_dict = {"x": -2147483648}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large positive 64-bit integer
    input_dict = {"x": 9223372036854775807}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: large negative 64-bit integer
    input_dict = {"x": -9223372036854775808}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isrealobj_2"] = isrealobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isrealobj_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isrealobj_2'.")


check_valid('jax.numpy.isrealobj', generated_inputs['jax.numpy.isrealobj_2'], lib="jax", suffix=2)
