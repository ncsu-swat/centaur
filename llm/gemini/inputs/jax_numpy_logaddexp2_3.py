
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logaddexp2_inputs():
    list_of_inputs = []

    # Input 1: Zero values
    input_dict = {"x1": 0, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small positive integers
    input_dict = {"x1": 1, "x2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integers
    input_dict = {"x1": -1, "x2": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mixed positive and negative integers
    input_dict = {"x1": 10, "x2": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive integers
    input_dict = {"x1": 100, "x2": 50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large negative and zero
    input_dict = {"x1": -100, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Identical positive integers
    input_dict = {"x1": 42, "x2": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Symmetric negative and positive integers
    input_dict = {"x1": -5, "x2": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large equal integers
    input_dict = {"x1": 1024, "x2": 1024}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Equal negative integers
    input_dict = {"x1": -50, "x2": -50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Zero and positive integer
    input_dict = {"x1": 0, "x2": 15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative and large positive integer
    input_dict = {"x1": -500, "x2": 500}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.logaddexp2_3"] = logaddexp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logaddexp2_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logaddexp2_3'.")


check_valid('jax.numpy.logaddexp2', generated_inputs['jax.numpy.logaddexp2_3'], lib="jax", suffix=3)
