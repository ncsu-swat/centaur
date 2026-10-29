
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_cdf_inputs():
    list_of_inputs = []

    # Input 1: Standard Cauchy evaluated at center
    input_dict = {"x": 0.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: One scale above center
    input_dict = {"x": 1.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: One scale below center
    input_dict = {"x": -1.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive shift and small scale
    input_dict = {"x": 5.5, "loc": 2.0, "scale": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative shift and larger scale
    input_dict = {"x": -10.0, "loc": -5.0, "scale": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive tail
    input_dict = {"x": 100.0, "loc": 0.0, "scale": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large negative tail
    input_dict = {"x": -100.0, "loc": 0.0, "scale": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small scale parameter
    input_dict = {"x": 10.005, "loc": 10.0, "scale": 0.01}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Floating point values with arbitrary decimals
    input_dict = {"x": 3.14159, "loc": 2.71828, "scale": 1.41421}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Extreme positive tail
    input_dict = {"x": 1e6, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Extreme negative tail
    input_dict = {"x": -1e6, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Very large scale parameter
    input_dict = {"x": 0.5, "loc": -0.5, "scale": 500.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.cdf_2"] = cauchy_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.cdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.cdf_2'.")


check_valid('jax.scipy.stats.cauchy.cdf', generated_inputs['jax.scipy.stats.cauchy.cdf_2'], lib="jax", suffix=2)
