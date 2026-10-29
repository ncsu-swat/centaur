
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def exp2_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive integer as float
    input_dict = {"x": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer as float
    input_dict = {"x": -2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive fractional value
    input_dict = {"x": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative fractional value
    input_dict = {"x": -1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Non-integer float (pi)
    input_dict = {"x": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small positive float
    input_dict = {"x": 1e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Moderately large positive float
    input_dict = {"x": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Moderately large negative float
    input_dict = {"x": -10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Special float - infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Special float - negative infinity
    input_dict = {"x": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Special float - nan
    input_dict = {"x": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.exp2_3"] = exp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.exp2_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.exp2_3'.")


check_valid('jax.numpy.exp2', generated_inputs['jax.numpy.exp2_3'], lib="jax", suffix=3)
