
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def result_type_inputs():
    list_of_inputs = []

    # Input 1: Simple positive integer
    input_dict = {"args": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero
    input_dict = {"args": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Simple negative integer
    input_dict = {"args": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive integer
    input_dict = {"args": 100000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative integer
    input_dict = {"args": -100000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer value 42
    input_dict = {"args": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Integer value -42
    input_dict = {"args": -42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Integer value 256
    input_dict = {"args": 256}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Integer value -256
    input_dict = {"args": -256}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Maximum 16-bit integer representation
    input_dict = {"args": 65535}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.result_type_4"] = result_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.result_type_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.result_type_4'.")


check_valid('jax.numpy.result_type', generated_inputs['jax.numpy.result_type_4'], lib="jax", suffix=4)
