
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def bitwise_not_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive small integer
    input_dict = {"x": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative small integer
    input_dict = {"x": int(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive integer (power of 2 minus 1)
    input_dict = {"x": int(7)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative integer
    input_dict = {"x": int(-8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 8-bit boundary positive value
    input_dict = {"x": int(127)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 8-bit boundary negative value
    input_dict = {"x": int(-128)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 8-bit unsigned boundary value
    input_dict = {"x": int(255)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Moderate positive integer
    input_dict = {"x": int(1024)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 16-bit boundary negative value
    input_dict = {"x": int(-32768)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 16-bit boundary positive value
    input_dict = {"x": int(65535)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Large positive integer
    input_dict = {"x": int(1000000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.bitwise_not_2"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_not_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_not_2'.")


check_valid('jax.lax.bitwise_not', generated_inputs['jax.lax.bitwise_not_2'], lib="jax", suffix=2)
