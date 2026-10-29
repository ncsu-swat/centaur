
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logaddexp2_inputs():
    list_of_inputs = []

    # Input 1: Zero values
    input_dict = {"x1": 0.0, "x2": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive floats
    input_dict = {"x1": 1.0, "x2": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative floats
    input_dict = {"x1": -1.0, "x2": -2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed positive and negative floats
    input_dict = {"x1": 3.5, "x2": -2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive floats (avoiding overflow)
    input_dict = {"x1": 1000.0, "x2": 999.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large negative floats (avoiding underflow)
    input_dict = {"x1": -1000.0, "x2": -1000.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small magnitude floats
    input_dict = {"x1": 1e-7, "x2": 2e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Distinct fractional values
    input_dict = {"x1": 0.125, "x2": 0.875}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Widely separated values
    input_dict = {"x1": 100.0, "x2": -100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-integer mathematical constants as floats
    input_dict = {"x1": 3.141592653589793, "x2": 2.718281828459045}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative infinity edge case
    input_dict = {"x1": float("-inf"), "x2": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.logaddexp2_2"] = logaddexp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logaddexp2_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logaddexp2_2'.")


check_valid('jax.numpy.logaddexp2', generated_inputs['jax.numpy.logaddexp2_2'], lib="jax", suffix=2)
