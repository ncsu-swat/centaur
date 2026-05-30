
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def positive_inputs():
    list_of_inputs = []

    # Input 1: Simple positive float
    input_dict = {"x": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple negative float
    input_dict = {"x": -3.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero float
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative zero float
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive float
    input_dict = {"x": 1e10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small negative float
    input_dict = {"x": -1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Pi approximation float
    input_dict = {"x": 3.1415926535}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Euler's number approximation negative float
    input_dict = {"x": -2.718281828}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Floating point infinity
    input_dict = {"x": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Floating point negative infinity
    input_dict = {"x": float('-inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: NaN float
    input_dict = {"x": float('nan')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.positive_3"] = positive_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.positive_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.positive_3'.")


check_valid('jax.numpy.positive', generated_inputs['jax.numpy.positive_3'], lib="jax", suffix=3)
