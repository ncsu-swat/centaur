
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isscalar_inputs():
    list_of_inputs = []

    # Input 1: Empty list
    input_dict = {"element": []}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List with one integer
    input_dict = {"element": [42]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List with one negative float
    input_dict = {"element": [-3.14]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List with multiple integers
    input_dict = {"element": [1, 2, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: List with multiple floats
    input_dict = {"element": [1.1, 2.2, 3.3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D Nested list (matrix-like)
    input_dict = {"element": [[1, 2], [3, 4]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List with a boolean
    input_dict = {"element": [True]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List with a complex number
    input_dict = {"element": [1 + 2j]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List with numpy scalar type
    input_dict = {"element": [np.int32(10)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: List with numpy float scalar
    input_dict = {"element": [np.float64(-0.001)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D Nested list
    input_dict = {"element": [[[1]]]}
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
