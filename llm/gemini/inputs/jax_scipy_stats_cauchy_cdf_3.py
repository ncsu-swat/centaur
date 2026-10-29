
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_cdf_inputs():
    list_of_inputs = []

    # Input 1: Standard values at median
    input_dict = {"x": int(0), "loc": int(0), "scale": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive x, standard loc and scale
    input_dict = {"x": int(1), "loc": int(0), "scale": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x, standard loc and scale
    input_dict = {"x": int(-1), "loc": int(0), "scale": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Shifted loc and scaled
    input_dict = {"x": int(5), "loc": int(2), "scale": int(3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative loc and negative x
    input_dict = {"x": int(-10), "loc": int(-5), "scale": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger scale and larger positive x
    input_dict = {"x": int(100), "loc": int(0), "scale": int(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Far left tail
    input_dict = {"x": int(-50), "loc": int(50), "scale": int(25)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero x with negative loc
    input_dict = {"x": int(0), "loc": int(-10), "scale": int(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x equals loc with unit scale
    input_dict = {"x": int(7), "loc": int(7), "scale": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Far negative tail with large scale
    input_dict = {"x": int(-1000), "loc": int(0), "scale": int(100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: All parameters identical positive integers
    input_dict = {"x": int(42), "loc": int(42), "scale": int(42)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.cdf_3"] = cauchy_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.cdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.cdf_3'.")


check_valid('jax.scipy.stats.cauchy.cdf', generated_inputs['jax.scipy.stats.cauchy.cdf_3'], lib="jax", suffix=3)
