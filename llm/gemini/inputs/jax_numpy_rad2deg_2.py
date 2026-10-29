
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rad2deg_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": float(0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Pi (180 degrees)
    input_dict = {"x": float(np.pi)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative Pi (-180 degrees)
    input_dict = {"x": float(-np.pi)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Pi / 2 (90 degrees)
    input_dict = {"x": float(np.pi / 2.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: -Pi / 4 (-45 degrees)
    input_dict = {"x": float(-np.pi / 4.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2 * Pi (360 degrees)
    input_dict = {"x": float(2.0 * np.pi)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1 radian (~57.2958 degrees)
    input_dict = {"x": float(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative arbitrary float
    input_dict = {"x": float(-3.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small positive float
    input_dict = {"x": float(1e-4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large positive float
    input_dict = {"x": float(100.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3 * Pi / 2 (270 degrees)
    input_dict = {"x": float(3.0 * np.pi / 2.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.rad2deg_2"] = rad2deg_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.rad2deg_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.rad2deg_2'.")


check_valid('jax.numpy.rad2deg', generated_inputs['jax.numpy.rad2deg_2'], lib="jax", suffix=2)
