
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def radians_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive 180 degrees
    input_dict = {"x": 180.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative 180 degrees
    input_dict = {"x": -180.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive 90 degrees
    input_dict = {"x": 90.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative 45 degrees
    input_dict = {"x": -45.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Positive 360 degrees
    input_dict = {"x": 360.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Approx one radian in degrees
    input_dict = {"x": 57.29577951308232}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small negative value
    input_dict = {"x": -0.01}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very small positive value
    input_dict = {"x": 1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large positive value
    input_dict = {"x": 720.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.radians_4"] = radians_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.radians_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.radians_4'.")


check_valid('jax.numpy.radians', generated_inputs['jax.numpy.radians_4'], lib="jax", suffix=4)
