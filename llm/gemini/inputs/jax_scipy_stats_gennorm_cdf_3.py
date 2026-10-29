
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gennorm_cdf_inputs():
    list_of_inputs = []

    # Input 1: standard normal equivalent (x=0, beta=2)
    input_dict = {"x": int(0), "beta": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Laplace equivalent with positive x (x=1, beta=1)
    input_dict = {"x": int(1), "beta": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Laplace equivalent with negative x (x=-1, beta=1)
    input_dict = {"x": int(-1), "beta": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive x, beta=2
    input_dict = {"x": int(2), "beta": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative x, beta=2
    input_dict = {"x": int(-2), "beta": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x=0, beta=1
    input_dict = {"x": int(0), "beta": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: larger beta with positive x
    input_dict = {"x": int(3), "beta": int(4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: negative x with odd beta
    input_dict = {"x": int(-5), "beta": int(3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: larger positive x and beta
    input_dict = {"x": int(10), "beta": int(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large negative x
    input_dict = {"x": int(-10), "beta": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: x=0 with large beta
    input_dict = {"x": int(0), "beta": int(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: positive x with beta=1
    input_dict = {"x": int(4), "beta": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.gennorm.cdf_3"] = gennorm_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.cdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.cdf_3'.")


check_valid('jax.scipy.stats.gennorm.cdf', generated_inputs['jax.scipy.stats.gennorm.cdf_3'], lib="jax", suffix=3)
