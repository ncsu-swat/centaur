
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def less_inputs():
    list_of_inputs = []

    # Input 1: positive integers, x < y
    input_dict = {"x": 3, "y": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive integers, x > y
    input_dict = {"x": 10, "y": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: equal positive integers
    input_dict = {"x": 42, "y": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: negative integers, x < y
    input_dict = {"x": -10, "y": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative integers, x > y
    input_dict = {"x": -3, "y": -8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: negative and positive integer
    input_dict = {"x": -1, "y": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: positive and negative integer
    input_dict = {"x": 15, "y": -20}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: zero and positive integer
    input_dict = {"x": 0, "y": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: zero and negative integer
    input_dict = {"x": 0, "y": -50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: both zeros
    input_dict = {"x": 0, "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: large positive integers
    input_dict = {"x": 1000000, "y": 2000000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: large negative integers
    input_dict = {"x": -5000000, "y": -1000000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.less_7"] = less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_7'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_7'], lib="jax", suffix=7)
