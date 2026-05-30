
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def deg2rad_inputs():
    list_of_inputs = []

    # Input 1: Zero degrees
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Right angle (90 degrees)
    input_dict = {"x": 90.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Straight angle (180 degrees)
    input_dict = {"x": 180.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Full rotation (360 degrees)
    input_dict = {"x": 360.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative acute angle
    input_dict = {"x": -45.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative straight angle
    input_dict = {"x": -180.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Approximately one radian in degrees
    input_dict = {"x": 57.29577951308232}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Double rotation
    input_dict = {"x": 720.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small positive angle
    input_dict = {"x": 0.001}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative fractional angle
    input_dict = {"x": -0.1}
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
