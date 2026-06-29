
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gennorm_cdf_inputs():
    list_of_inputs = []

    # Input 1: standard normal case (beta = 2, which corresponds to standard normal-like shape)
    input_dict = {
        "x": 0.0,
        "beta": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Laplace-like shape (beta = 1) with positive x
    input_dict = {
        "x": 1.0,
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Laplace-like shape (beta = 1) with negative x
    input_dict = {
        "x": -1.0,
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Heavy-tailed shape (beta < 1) with positive x
    input_dict = {
        "x": 0.5,
        "beta": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Heavy-tailed shape (beta < 1) with negative x
    input_dict = {
        "x": -0.5,
        "beta": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Light-tailed shape (beta > 2) with positive x
    input_dict = {
        "x": 2.5,
        "beta": 5.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Light-tailed shape (beta > 2) with negative x
    input_dict = {
        "x": -2.5,
        "beta": 5.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small beta value
    input_dict = {
        "x": 0.1,
        "beta": 0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large beta value
    input_dict = {
        "x": 1.5,
        "beta": 10.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large positive x, beta = 2.0
    input_dict = {
        "x": 10.0,
        "beta": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Large negative x, beta = 2.0
    input_dict = {
        "x": -10.0,
        "beta": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gennorm.cdf_2"] = gennorm_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.cdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.cdf_2'.")


check_valid('jax.scipy.stats.gennorm.cdf', generated_inputs['jax.scipy.stats.gennorm.cdf_2'], lib="jax", suffix=2)
