
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def square_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive unit integer
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative unit integer
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive integer
    input_dict = {"x": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small negative integer
    input_dict = {"x": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Moderate positive integer
    input_dict = {"x": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Moderate negative integer
    input_dict = {"x": -16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger positive integer
    input_dict = {"x": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger negative integer
    input_dict = {"x": -250}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Odd positive integer
    input_dict = {"x": 37}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative integer
    input_dict = {"x": -42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Big integer within standard 32-bit range
    input_dict = {"x": 1024}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.square_2"] = square_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.square_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.square_2'.")


check_valid('jax.numpy.square', generated_inputs['jax.numpy.square_2'], lib="jax", suffix=2)
