
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_pdf_inputs():
    list_of_inputs = []

    # Input 1: standard normal at mean
    input_dict = {"x": 0, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive x with standard normal
    input_dict = {"x": 1, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative x with standard normal
    input_dict = {"x": -1, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive mean and larger scale
    input_dict = {"x": 2, "loc": 1, "scale": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative mean and negative x
    input_dict = {"x": -3, "loc": -1, "scale": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x at the mean with scale > 1
    input_dict = {"x": 5, "loc": 5, "scale": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large negative x
    input_dict = {"x": -10, "loc": 0, "scale": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: zero x with negative mean
    input_dict = {"x": 0, "loc": -5, "scale": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x matching loc with unit scale
    input_dict = {"x": 4, "loc": 4, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large values for x, loc, and scale
    input_dict = {"x": 100, "loc": 50, "scale": 20}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: negative loc matching x
    input_dict = {"x": -20, "loc": -20, "scale": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.norm.pdf_3"] = norm_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.pdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.pdf_3'.")


check_valid('jax.scipy.stats.norm.pdf', generated_inputs['jax.scipy.stats.norm.pdf_3'], lib="jax", suffix=3)
