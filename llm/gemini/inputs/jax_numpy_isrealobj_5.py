
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isrealobj_inputs():
    list_of_inputs = []

    # Input 1: List of integers
    input_dict = {"x": [1, 2, -3, 4]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: List of floats
    input_dict = {"x": [1.0, 2.5, -3.2, 0.0]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: List of complex numbers
    input_dict = {"x": [1+2j, 3-4j, 0+0j]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: List of complex numbers with zero imaginary parts
    input_dict = {"x": [1+0j, -2.5+0j]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Nested list of floats (2D structure)
    input_dict = {"x": [[1.0, 2.0], [3.0, 4.0]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Nested list containing complex numbers
    input_dict = {"x": [[1.0, 2j], [3.0, 4.0]]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: List of numpy arrays (real)
    input_dict = {"x": [np.array([1, 2]), np.array([3, 4])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: List of numpy arrays (complex)
    input_dict = {"x": [np.array([1+2j, 2]), np.array([3, 4])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: List of booleans
    input_dict = {"x": [True, False, True]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: List of mixed types (int and float)
    input_dict = {"x": [1, 2.0, 3, 4.5]}
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
