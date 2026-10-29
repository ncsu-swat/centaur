
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isrealobj_inputs():
    list_of_inputs = []

    # Input 1: 1D list of positive integers
    input_dict = {"x": [1, 2, 3, 4]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D list of negative and zero integers
    input_dict = {"x": [-10, -5, 0, 5]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D list of floats
    input_dict = {"x": [1.5, -2.3, 0.0, 4.7]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D list of booleans
    input_dict = {"x": [True, False, True, True]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D list of complex numbers with non-zero imaginary parts
    input_dict = {"x": [1.0 + 2.0j, -3.0 + 4.0j]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D list of complex numbers with zero imaginary parts
    input_dict = {"x": [1.0 + 0.0j, 2.0 + 0.0j, -3.0 + 0.0j]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D nested list of integers
    input_dict = {"x": [[1, 2, 3], [4, 5, 6]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D nested list of floats
    input_dict = {"x": [[-1.2, 0.5], [3.3, -4.1]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D nested list of complex numbers
    input_dict = {"x": [[1j, 2j], [3 - 1j, 4 + 0j]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D nested list of floats
    input_dict = {"x": [[[1.0, -2.0], [3.0, 4.0]], [[5.0, 6.0], [-7.0, 8.0]]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty list
    input_dict = {"x": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Mixed int and float list
    input_dict = {"x": [1, 2.5, -3, 4.2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isrealobj_5"] = isrealobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isrealobj_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isrealobj_5'.")


check_valid('jax.numpy.isrealobj', generated_inputs['jax.numpy.isrealobj_5'], lib="jax", suffix=5)
