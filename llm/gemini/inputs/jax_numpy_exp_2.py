
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def exp_inputs():
    list_of_inputs = []

    # Input 1: positive float
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative float
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: large positive float
    input_dict = {"x": 50.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: large negative float
    input_dict = {"x": -50.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: small positive float
    input_dict = {"x": 1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: small negative float
    input_dict = {"x": -1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: mathematical constant pi
    input_dict = {"x": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: infinity
    input_dict = {"x": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: negative infinity
    input_dict = {"x": float('-inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: NaN
    input_dict = {"x": float('nan')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.exp_2"] = exp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.exp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.exp_2'.")


check_valid('jax.numpy.exp', generated_inputs['jax.numpy.exp_2'], lib="jax", suffix=2)
