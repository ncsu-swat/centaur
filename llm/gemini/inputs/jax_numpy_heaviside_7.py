
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: Positive float and positive integer
    list_of_inputs.append(copy.deepcopy({"x1": 1.5, "x2": 2}))

    # Input 2: Zero float and zero integer
    list_of_inputs.append(copy.deepcopy({"x1": 0.0, "x2": 0}))

    # Input 3: Negative float and positive integer
    list_of_inputs.append(copy.deepcopy({"x1": -2.5, "x2": 1}))

    # Input 4: Positive float and negative integer
    list_of_inputs.append(copy.deepcopy({"x1": 0.5, "x2": -1}))

    # Input 5: Large positive float and large integer
    list_of_inputs.append(copy.deepcopy({"x1": 1000.0, "x2": 42}))

    # Input 6: Large negative float and large negative integer
    list_of_inputs.append(copy.deepcopy({"x1": -1000.0, "x2": -42}))

    # Input 7: Float zero and positive integer
    list_of_inputs.append(copy.deepcopy({"x1": 0.0, "x2": 5}))

    # Input 8: Negative float and negative integer
    list_of_inputs.append(copy.deepcopy({"x1": -3.14, "x2": -2}))

    # Input 9: Small positive float and zero integer
    list_of_inputs.append(copy.deepcopy({"x1": 1e-5, "x2": 0}))

    # Input 10: Small negative float and positive integer
    list_of_inputs.append(copy.deepcopy({"x1": -1e-5, "x2": 10}))

    return list_of_inputs

generated_inputs["jax.numpy.heaviside_7"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_7'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_7'], lib="jax", suffix=7)
