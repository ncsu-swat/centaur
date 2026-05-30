
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def round_inputs():
    list_of_inputs = []

    # Input 1: Basic positive float, decimals = 0
    input_dict = {"a": 1.532, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive float with decimals = 2
    input_dict = {"a": 3.267, "decimals": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float, decimals = 1
    input_dict = {"a": -6.149, "decimals": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Halfway value, decimals = 0 (rounds to even)
    input_dict = {"a": 10.5, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative halfway value, decimals = 0
    input_dict = {"a": -10.5, "decimals": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small float, decimals = 5
    input_dict = {"a": 0.123456789, "decimals": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large float, decimals = 3
    input_dict = {"a": 12345.6789, "decimals": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative small float, decimals = 4
    input_dict = {"a": -0.000456, "decimals": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Floating point value close to next integer, decimals = 1
    input_dict = {"a": 99.999, "decimals": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High precision float, decimals = 6
    input_dict = {"a": 2.7182818284, "decimals": 6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.round_2"] = round_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.round_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.round_2'.")


check_valid('jax.numpy.round', generated_inputs['jax.numpy.round_2'], lib="jax", suffix=2)
