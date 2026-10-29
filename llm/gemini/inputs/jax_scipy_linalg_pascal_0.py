
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pascal_inputs():
    list_of_inputs = []

    # Input 1: size 1, symmetric
    input_dict = {"n": 1, "kind": "symmetric"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: size 1, lower
    input_dict = {"n": 1, "kind": "lower"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: size 1, upper
    input_dict = {"n": 1, "kind": "upper"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: size 2, symmetric
    input_dict = {"n": 2, "kind": "symmetric"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: size 3, lower
    input_dict = {"n": 3, "kind": "lower"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: size 4, upper
    input_dict = {"n": 4, "kind": "upper"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: size 5, symmetric
    input_dict = {"n": 5, "kind": "symmetric"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: size 6, lower
    input_dict = {"n": 6, "kind": "lower"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: size 7, upper
    input_dict = {"n": 7, "kind": "upper"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: size 10, symmetric
    input_dict = {"n": 10, "kind": "symmetric"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: size 12, lower
    input_dict = {"n": 12, "kind": "lower"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: size 15, upper
    input_dict = {"n": 15, "kind": "upper"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.pascal"] = pascal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.pascal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.pascal'.")


check_valid('jax.scipy.linalg.pascal', generated_inputs['jax.scipy.linalg.pascal'], lib="jax", suffix=0)
