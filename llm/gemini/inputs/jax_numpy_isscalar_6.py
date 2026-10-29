
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isscalar_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    input_dict = {"element": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single-element list with integer
    input_dict = {"element": [0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single-element list with negative integer
    input_dict = {"element": [-42]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of floats
    input_dict = {"element": [1.5, -2.5, 3.14]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Nested 2D list of integers
    input_dict = {"element": [[1, 2], [3, 4]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Nested 2D list of floats with negative values
    input_dict = {"element": [[-1.0, 0.0], [0.5, -2.5]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D nested list
    input_dict = {"element": [[[1, 2]], [[3, 4]]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of booleans
    input_dict = {"element": [True, False, True]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List of complex numbers
    input_dict = {"element": [1 + 2j, -3 - 4j]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Nested list with a single element
    input_dict = {"element": [[1]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: List containing numpy scalar types
    input_dict = {"element": [np.float32(1.0), np.float32(-2.0)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isscalar_6"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_6'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_6'], lib="jax", suffix=6)
