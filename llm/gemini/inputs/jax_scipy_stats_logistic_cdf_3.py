
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def logistic_cdf_inputs():
    list_of_inputs = []

    # Input 1: Basic standard logistic at 0
    input_dict = {
        "x": int(0),
        "loc": int(0),
        "scale": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive values
    input_dict = {
        "x": int(5),
        "loc": int(2),
        "scale": int(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x
    input_dict = {
        "x": int(-5),
        "loc": int(0),
        "scale": int(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative loc
    input_dict = {
        "x": int(0),
        "loc": int(-3),
        "scale": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative x and negative loc
    input_dict = {
        "x": int(-10),
        "loc": int(-5),
        "scale": int(4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger scale
    input_dict = {
        "x": int(15),
        "loc": int(5),
        "scale": int(20)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x equal to loc
    input_dict = {
        "x": int(7),
        "loc": int(7),
        "scale": int(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger negative integer for x
    input_dict = {
        "x": int(-100),
        "loc": int(0),
        "scale": int(10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive integer for x
    input_dict = {
        "x": int(100),
        "loc": int(10),
        "scale": int(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scale is 1 with offset loc
    input_dict = {
        "x": int(4),
        "loc": int(-2),
        "scale": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Scale with moderate positive value
    input_dict = {
        "x": int(-1),
        "loc": int(1),
        "scale": int(8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.cdf_3"] = logistic_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.cdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.cdf_3'.")


check_valid('jax.scipy.stats.logistic.cdf', generated_inputs['jax.scipy.stats.logistic.cdf_3'], lib="jax", suffix=3)
