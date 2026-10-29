
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def power_inputs():
    list_of_inputs = []

    # Input 1: Basic positive base and exponent
    input_dict = {"x1": 2, "x2": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero exponent
    input_dict = {"x1": 5, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative base with even exponent
    input_dict = {"x1": -3, "x2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative base with odd exponent
    input_dict = {"x1": -2, "x2": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero base with positive exponent
    input_dict = {"x1": 0, "x2": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero base with zero exponent
    input_dict = {"x1": 0, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Base of 1 with larger exponent
    input_dict = {"x1": 1, "x2": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Base of -1 with odd exponent
    input_dict = {"x1": -1, "x2": 17}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Base of -1 with even exponent
    input_dict = {"x1": -1, "x2": 20}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger positive base and exponent
    input_dict = {"x1": 10, "x2": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Base 7 with exponent 1
    input_dict = {"x1": 7, "x2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.power_6"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_6'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_6'], lib="jax", suffix=6)
