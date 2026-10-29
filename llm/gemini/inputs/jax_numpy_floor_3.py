
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def floor_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    input_dict = {"x": 3.7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative float
    input_dict = {"x": -2.3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero float
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative zero float
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Exact integer as float
    input_dict = {"x": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Exact negative integer as float
    input_dict = {"x": -4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small positive float
    input_dict = {"x": 1e-6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small negative float
    input_dict = {"x": -1e-6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive float
    input_dict = {"x": 1234567.89}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large negative float
    input_dict = {"x": -987654.321}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative infinity
    input_dict = {"x": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: NaN
    input_dict = {"x": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.floor_3"] = floor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_3'.")


check_valid('jax.numpy.floor', generated_inputs['jax.numpy.floor_3'], lib="jax", suffix=3)
