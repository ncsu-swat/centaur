
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


def bessel_i1e_inputs():
    list_of_inputs = []

    # Input 1: zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive small value
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative small value
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive value
    input_dict = {"x": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative value
    input_dict = {"x": -2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: medium positive value
    input_dict = {"x": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: medium negative value
    input_dict = {"x": -5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: larger positive value
    input_dict = {"x": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: larger negative value
    input_dict = {"x": -10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large positive value
    input_dict = {"x": 50.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: large negative value
    input_dict = {"x": -50.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: very large positive value
    input_dict = {"x": 100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.bessel_i1e_3"] = bessel_i1e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i1e_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i1e_3'.")


check_valid('jax.lax.bessel_i1e', generated_inputs['jax.lax.bessel_i1e_3'], lib="jax", suffix=3)
