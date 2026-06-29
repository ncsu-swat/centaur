
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_logistic_cdf_inputs():
    list_of_inputs = []

    # Input 1: Standard logistic at 0
    input_dict = {
        "x": 0.0,
        "loc": 0.0,
        "scale": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive x, standard loc/scale
    input_dict = {
        "x": 1.5,
        "loc": 0.0,
        "scale": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x, positive loc, scale > 1
    input_dict = {
        "x": -2.5,
        "loc": 1.0,
        "scale": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive x, negative loc, small scale
    input_dict = {
        "x": 10.0,
        "loc": -5.0,
        "scale": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative x, positive loc, scale > 1
    input_dict = {
        "x": -10.0,
        "loc": 5.0,
        "scale": 3.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x equals loc, very small scale
    input_dict = {
        "x": 0.5,
        "loc": 0.5,
        "scale": 0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very large scale
    input_dict = {
        "x": 100.0,
        "loc": 0.0,
        "scale": 10.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative loc, large scale
    input_dict = {
        "x": -100.0,
        "loc": -10.0,
        "scale": 50.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Fractional values
    input_dict = {
        "x": 0.0,
        "loc": 1.23,
        "scale": 4.56
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Standard test case with negative scale shift
    input_dict = {
        "x": -0.75,
        "loc": -2.5,
        "scale": 0.75
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.cdf_2"] = jax_scipy_stats_logistic_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.cdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.cdf_2'.")


check_valid('jax.scipy.stats.logistic.cdf', generated_inputs['jax.scipy.stats.logistic.cdf_2'], lib="jax", suffix=2)
