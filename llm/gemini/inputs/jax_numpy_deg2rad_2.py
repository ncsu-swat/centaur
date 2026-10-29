
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def deg2rad_inputs():
    list_of_inputs = []

    # Input 1: zero degrees
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: right angle (90 degrees)
    input_dict = {"x": 90.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: straight angle (180 degrees)
    input_dict = {"x": 180.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: full circle (360 degrees)
    input_dict = {"x": 360.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative angle (-45 degrees)
    input_dict = {"x": -45.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: negative full rotation (-360 degrees)
    input_dict = {"x": -360.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: fractional degree
    input_dict = {"x": 33.3333}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: small fractional angle
    input_dict = {"x": 1e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large angle
    input_dict = {"x": 7200.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: negative fractional angle
    input_dict = {"x": -123.456}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: NaN value
    input_dict = {"x": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.deg2rad_2"] = deg2rad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.deg2rad_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.deg2rad_2'.")


check_valid('jax.numpy.deg2rad', generated_inputs['jax.numpy.deg2rad_2'], lib="jax", suffix=2)
