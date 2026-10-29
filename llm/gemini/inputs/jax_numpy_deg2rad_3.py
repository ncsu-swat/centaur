
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def deg2rad_inputs():
    list_of_inputs = []

    # Input 1: Zero degrees
    list_of_inputs.append({"x": int(0)})

    # Input 2: Right angle (positive)
    list_of_inputs.append({"x": int(90)})

    # Input 3: Straight angle
    list_of_inputs.append({"x": int(180)})

    # Input 4: Full circle
    list_of_inputs.append({"x": int(360)})

    # Input 5: Negative right angle
    list_of_inputs.append({"x": int(-90)})

    # Input 6: Negative straight angle
    list_of_inputs.append({"x": int(-180)})

    # Input 7: Acute angle
    list_of_inputs.append({"x": int(45)})

    # Input 8: Acute angle (30 degrees)
    list_of_inputs.append({"x": int(30)})

    # Input 9: Large angle (multiple rotations)
    list_of_inputs.append({"x": int(720)})

    # Input 10: Negative large angle
    list_of_inputs.append({"x": int(-360)})

    # Input 11: Three-quarter circle
    list_of_inputs.append({"x": int(270)})

    # Input 12: Negative acute angle
    list_of_inputs.append({"x": int(-45)})

    return list_of_inputs


generated_inputs["jax.numpy.deg2rad_3"] = deg2rad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.deg2rad_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.deg2rad_3'.")


check_valid('jax.numpy.deg2rad', generated_inputs['jax.numpy.deg2rad_3'], lib="jax", suffix=3)
