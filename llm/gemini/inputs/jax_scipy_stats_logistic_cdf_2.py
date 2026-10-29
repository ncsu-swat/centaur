
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def logistic_cdf_inputs():
    list_of_inputs = []

    # Input 1: Standard logistic cdf at 0.0
    input_dict = {
        "x": 0.0,
        "loc": 0.0,
        "scale": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive x with standard loc/scale
    input_dict = {
        "x": 2.5,
        "loc": 0.0,
        "scale": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x
    input_dict = {
        "x": -3.0,
        "loc": 0.0,
        "scale": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Shifted location
    input_dict = {
        "x": 5.0,
        "loc": 5.0,
        "scale": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Shifted location and scaled
    input_dict = {
        "x": 10.0,
        "loc": 2.0,
        "scale": 4.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative location
    input_dict = {
        "x": -1.5,
        "loc": -2.0,
        "scale": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small scale parameter
    input_dict = {
        "x": 0.01,
        "loc": 0.0,
        "scale": 0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large scale parameter
    input_dict = {
        "x": 15.2,
        "loc": 1.0,
        "scale": 50.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large negative x
    input_dict = {
        "x": -25.0,
        "loc": 0.0,
        "scale": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-integer floats
    input_dict = {
        "x": 0.732,
        "loc": -0.456,
        "scale": 1.234
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Very small positive scale
    input_dict = {
        "x": -0.05,
        "loc": 0.0,
        "scale": 0.001
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.cdf_2"] = logistic_cdf_inputs()

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
