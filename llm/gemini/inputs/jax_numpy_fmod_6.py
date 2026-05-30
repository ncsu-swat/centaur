
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmod_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats
    input_dict = {"x1": 5.0, "x2": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative dividend
    input_dict = {"x1": -5.0, "x2": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative divisor
    input_dict = {"x1": 5.0, "x2": -2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Both negative
    input_dict = {"x1": -5.0, "x2": -2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Decimal values
    input_dict = {"x1": 1.5, "x2": 0.4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Divisible values
    input_dict = {"x1": 9.0, "x2": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large float dividend
    input_dict = {"x1": 1000000.0, "x2": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero dividend
    input_dict = {"x1": 0.0, "x2": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Floating point values (pi approximation)
    input_dict = {"x1": 3.14159, "x2": 1.57}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Very small floats
    input_dict = {"x1": 1.234e-5, "x2": 1.0e-6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fmod_6"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_6'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_6'], lib="jax", suffix=6)
