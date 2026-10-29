
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def negative_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive integer
    input_dict = {"x": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small negative integer
    input_dict = {"x": int(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Moderate positive integer
    input_dict = {"x": int(42)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Moderate negative integer
    input_dict = {"x": int(-42)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 8-bit positive limit
    input_dict = {"x": int(127)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 8-bit negative limit
    input_dict = {"x": int(-128)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 16-bit positive boundary
    input_dict = {"x": int(32767)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 16-bit negative boundary
    input_dict = {"x": int(-32768)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large positive integer
    input_dict = {"x": int(1000000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Large negative integer
    input_dict = {"x": int(-1000000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 32-bit positive limit
    input_dict = {"x": int(2147483647)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.negative_2"] = negative_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.negative_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.negative_2'.")


check_valid('jax.numpy.negative', generated_inputs['jax.numpy.negative_2'], lib="jax", suffix=2)
