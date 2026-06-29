
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_logistic_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Standard logistic distribution parameters
    input_dict = {
        'x': 0.0,
        'loc': 0.0,
        'scale': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive x, default loc and scale
    input_dict = {
        'x': 1.5,
        'loc': 0.0,
        'scale': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x, default loc and scale
    input_dict = {
        'x': -2.5,
        'loc': 0.0,
        'scale': 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive loc, scale > 1
    input_dict = {
        'x': 2.0,
        'loc': 1.0,
        'scale': 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative loc, scale < 1
    input_dict = {
        'x': -1.0,
        'loc': -2.0,
        'scale': 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large x, large scale
    input_dict = {
        'x': 10.0,
        'loc': 0.0,
        'scale': 5.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small x, small scale
    input_dict = {
        'x': 0.1,
        'loc': 0.0,
        'scale': 0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x equals loc, scale > 1
    input_dict = {
        'x': 3.5,
        'loc': 3.5,
        'scale': 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Extreme positive value
    input_dict = {
        'x': 100.0,
        'loc': 10.0,
        'scale': 20.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Extreme negative value
    input_dict = {
        'x': -100.0,
        'loc': -10.0,
        'scale': 20.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.logpdf_3"] = jax_scipy_stats_logistic_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.logpdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.logpdf_3'.")


check_valid('jax.scipy.stats.logistic.logpdf', generated_inputs['jax.scipy.stats.logistic.logpdf_3'], lib="jax", suffix=3)
