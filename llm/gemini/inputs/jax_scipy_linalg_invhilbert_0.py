
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def invhilbert_inputs():
    list_of_inputs = []

    # Input 1: order 1 matrix, float approximation
    input_dict = {"n": 1, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: order 2 matrix, exact integer computation
    input_dict = {"n": 2, "exact": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: order 3 matrix, float approximation
    input_dict = {"n": 3, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: order 4 matrix, exact integer computation
    input_dict = {"n": 4, "exact": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: order 5 matrix, float approximation
    input_dict = {"n": 5, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: order 6 matrix, exact integer computation
    input_dict = {"n": 6, "exact": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: order 7 matrix, float approximation
    input_dict = {"n": 7, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: order 8 matrix, exact integer computation
    input_dict = {"n": 8, "exact": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: order 9 matrix, float approximation
    input_dict = {"n": 9, "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: order 10 matrix, exact integer computation
    input_dict = {"n": 10, "exact": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.invhilbert"] = invhilbert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.invhilbert' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.invhilbert'.")


check_valid('jax.scipy.linalg.invhilbert', generated_inputs['jax.scipy.linalg.invhilbert'], lib="jax", suffix=0)
