
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_l_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Basic standard parameters
    input_dict = {"x": 0, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive x with standard loc and scale
    input_dict = {"x": 2, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x with larger scale
    input_dict = {"x": -3, "loc": 0, "scale": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Shifted location and scaled
    input_dict = {"x": 5, "loc": 2, "scale": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative location and negative x
    input_dict = {"x": -10, "loc": -5, "scale": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x equals location
    input_dict = {"x": 10, "loc": 10, "scale": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative x, positive location
    input_dict = {"x": -4, "loc": 4, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger values with large scale
    input_dict = {"x": 100, "loc": 50, "scale": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x at 0 with negative location
    input_dict = {"x": 0, "loc": -8, "scale": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Deep negative x value
    input_dict = {"x": -50, "loc": 0, "scale": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Scale greater than difference
    input_dict = {"x": 1, "loc": -1, "scale": 20}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gumbel_l.logcdf_3"] = gumbel_l_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.logcdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.logcdf_3'.")


check_valid('jax.scipy.stats.gumbel_l.logcdf', generated_inputs['jax.scipy.stats.gumbel_l.logcdf_3'], lib="jax", suffix=3)
