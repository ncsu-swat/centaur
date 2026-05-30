
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iterable_inputs():
    list_of_inputs = []

    # Input 1: Empty tuple
    input_dict = {"y": ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple of positive integers
    input_dict = {"y": (1, 2, 3, 4, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple of negative and positive integers
    input_dict = {"y": (-10, -5, 0, 5, 10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tuple of positive floats
    input_dict = {"y": (1.1, 2.2, 3.3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple of negative and positive floats
    input_dict = {"y": (-1.5, 0.0, 1.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tuple of booleans
    input_dict = {"y": (True, False, True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large range of integers
    input_dict = {"y": tuple(range(-50, 50))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element integer tuple
    input_dict = {"y": (42,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single element float tuple
    input_dict = {"y": (-3.14,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tuple of small-scale floats
    input_dict = {"y": (1e-5, 2e-5, 3e-5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Tuple of large-scale integers
    input_dict = {"y": (100000, 200000, 300000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.iterable_3"] = iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_3'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_3'], lib="jax", suffix=3)
