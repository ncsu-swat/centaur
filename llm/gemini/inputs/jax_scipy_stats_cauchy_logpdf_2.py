
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Standard Cauchy distribution at the mode
    input_dict = {"x": 0.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Point at x = 1.0
    input_dict = {"x": 1.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x value
    input_dict = {"x": -1.0, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Non-zero positive location and fractional scale
    input_dict = {"x": 5.5, "loc": 2.0, "scale": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative location and larger scale
    input_dict = {"x": -10.2, "loc": -5.0, "scale": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive value with scale 10.0
    input_dict = {"x": 100.0, "loc": 0.0, "scale": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small values near zero with small scale
    input_dict = {"x": -0.001, "loc": 0.0, "scale": 0.01}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Irrational approximation constants
    input_dict = {"x": 3.14159265, "loc": 2.71828182, "scale": 1.41421356}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large scale parameter
    input_dict = {"x": 0.5, "loc": 1.0, "scale": 100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tail evaluation with small scale
    input_dict = {"x": -50.0, "loc": 50.0, "scale": 0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Tiny scale parameter
    input_dict = {"x": 1e-5, "loc": 0.0, "scale": 1e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.logpdf_2"] = cauchy_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logpdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logpdf_2'.")


check_valid('jax.scipy.stats.cauchy.logpdf', generated_inputs['jax.scipy.stats.cauchy.logpdf_2'], lib="jax", suffix=2)
