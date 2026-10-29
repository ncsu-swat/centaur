
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def real_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"val": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive integer
    input_dict = {"val": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small negative integer
    input_dict = {"val": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Moderate positive integer
    input_dict = {"val": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Moderate negative integer
    input_dict = {"val": -100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive integer
    input_dict = {"val": 65536}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large negative integer
    input_dict = {"val": -123456}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large positive integer (near 32-bit limit)
    input_dict = {"val": 2147483647}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very large negative integer (near 32-bit limit)
    input_dict = {"val": -2147483648}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Odd positive integer
    input_dict = {"val": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Odd negative integer
    input_dict = {"val": -999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.real_2"] = real_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.real_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.real_2'.")


check_valid('jax.numpy.real', generated_inputs['jax.numpy.real_2'], lib="jax", suffix=2)
