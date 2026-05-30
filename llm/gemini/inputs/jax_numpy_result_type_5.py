
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def result_type_inputs():
    list_of_inputs = []

    # Input 1: positive float
    input_dict = {"args": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative float
    input_dict = {"args": -2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero
    input_dict = {"args": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: large float
    input_dict = {"args": 1e10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: small negative float
    input_dict = {"args": -1e-10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: pi approximation
    input_dict = {"args": 3.14159}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: infinity
    input_dict = {"args": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NaN
    input_dict = {"args": float('nan')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: negative zero
    input_dict = {"args": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: small scientific notation float
    input_dict = {"args": 1.234567e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.result_type_5"] = result_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.result_type_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.result_type_5'.")


check_valid('jax.numpy.result_type', generated_inputs['jax.numpy.result_type_5'], lib="jax", suffix=5)
