
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def iterable_inputs():
    list_of_inputs = []

    # Input 1: Empty tuple
    input_dict = {"y": ()}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Tuple with a single integer
    input_dict = {"y": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tuple of two positive integers
    input_dict = {"y": (1, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tuple of multiple integers
    input_dict = {"y": (1, 2, 3, 4, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tuple of integers with negative values and zero
    input_dict = {"y": (-10, -5, 0, 5, 10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tuple of floats
    input_dict = {"y": (1.0, 2.5, -3.7, 4.2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tuple of large integers
    input_dict = {"y": (100, 200, 300, 400)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tuple with a single negative float
    input_dict = {"y": (-1.0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tuple of zeros
    input_dict = {"y": (0.0, 0.0, 0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tuple of alternating sign integers
    input_dict = {"y": (1, -1, 2, -2, 3, -3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Tuple of scientific notation floats
    input_dict = {"y": (1e-5, 1e5, 0.5, -1e-2)}
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
