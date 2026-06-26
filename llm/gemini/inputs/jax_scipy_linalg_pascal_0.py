
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pascal_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({"n": 1, "kind": "symmetric"})

    # Input 2
    list_of_inputs.append({"n": 2, "kind": "lower"})

    # Input 3
    list_of_inputs.append({"n": 3, "kind": "upper"})

    # Input 4
    list_of_inputs.append({"n": 4, "kind": "symmetric"})

    # Input 5
    list_of_inputs.append({"n": 5, "kind": "lower"})

    # Input 6
    list_of_inputs.append({"n": 6, "kind": "upper"})

    # Input 7
    list_of_inputs.append({"n": 7, "kind": "symmetric"})

    # Input 8
    list_of_inputs.append({"n": 8, "kind": "lower"})

    # Input 9
    list_of_inputs.append({"n": 9, "kind": "upper"})

    # Input 10
    list_of_inputs.append({"n": 10, "kind": "symmetric"})

    # Input 11
    list_of_inputs.append({"n": 15, "kind": "lower"})

    # Input 12
    list_of_inputs.append({"n": 20, "kind": "upper"})

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
