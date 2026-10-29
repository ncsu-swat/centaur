
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isinf_inputs():
    list_of_inputs = []

    # Input 1: boolean True
    input_dict = {"x": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: boolean False
    input_dict = {"x": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: boolean True from comparison
    input_dict = {"x": bool(1 == 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: boolean False from comparison
    input_dict = {"x": bool(1 == 0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: boolean True
    input_dict = {"x": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: boolean False
    input_dict = {"x": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: boolean True cast from non-zero int
    input_dict = {"x": bool(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: boolean False cast from zero int
    input_dict = {"x": bool(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: boolean True
    input_dict = {"x": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: boolean False
    input_dict = {"x": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isinf_4"] = isinf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isinf_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isinf_4'.")


check_valid('jax.numpy.isinf', generated_inputs['jax.numpy.isinf_4'], lib="jax", suffix=4)
