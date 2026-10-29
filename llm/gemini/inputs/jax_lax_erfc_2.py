
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def erfc_inputs():
    list_of_inputs = []

    # Input 1: zero as float
    input_dict = {"x": float(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: small positive value
    input_dict = {"x": float(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: small negative value
    input_dict = {"x": float(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive value
    input_dict = {"x": float(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative value
    input_dict = {"x": float(-2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: medium positive value
    input_dict = {"x": float(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: medium negative value
    input_dict = {"x": float(-5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: larger positive value
    input_dict = {"x": float(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: larger negative value
    input_dict = {"x": float(-10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large positive value where erfc underflows to 0
    input_dict = {"x": float(25)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: large negative value where erfc saturates to 2
    input_dict = {"x": float(-25)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.erfc_2"] = erfc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erfc_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erfc_2'.")


check_valid('jax.lax.erfc', generated_inputs['jax.lax.erfc_2'], lib="jax", suffix=2)
