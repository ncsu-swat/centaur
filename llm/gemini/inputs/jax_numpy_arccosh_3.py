
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arccosh_inputs():
    list_of_inputs = []

    # Input 1: positive integer at domain boundary (arccosh(1) = 0)
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: small positive integer > 1
    input_dict = {"x": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: moderate positive integer
    input_dict = {"x": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: integer 10
    input_dict = {"x": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: integer zero (outside real domain, produces nan)
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: negative integer -1 (outside real domain, produces nan)
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: negative integer -5
    input_dict = {"x": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: negative integer -100
    input_dict = {"x": -100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: larger positive integer
    input_dict = {"x": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large positive integer
    input_dict = {"x": 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: positive integer 3
    input_dict = {"x": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.arccosh_3"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccosh_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccosh_3'.")


check_valid('jax.numpy.arccosh', generated_inputs['jax.numpy.arccosh_3'], lib="jax", suffix=3)
