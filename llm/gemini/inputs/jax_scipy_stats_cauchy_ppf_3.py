
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_ppf_inputs():
    list_of_inputs = []

    # Input 1: q=0 at standard location and scale
    input_dict = {"q": 0, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: q=1 at standard location and scale
    input_dict = {"q": 1, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: q=0 with positive location and scale
    input_dict = {"q": 0, "loc": 5, "scale": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: q=1 with negative location and scale
    input_dict = {"q": 1, "loc": -3, "scale": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: q=0 with large negative location
    input_dict = {"q": 0, "loc": -10, "scale": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: q=1 with large positive location and scale
    input_dict = {"q": 1, "loc": 100, "scale": 50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: q=0 with positive offset
    input_dict = {"q": 0, "loc": 1, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: q=1 with moderate location and scale
    input_dict = {"q": 1, "loc": 2, "scale": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: q=0 with negative location
    input_dict = {"q": 0, "loc": -1, "scale": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: q=1 with zero location and large scale
    input_dict = {"q": 1, "loc": 0, "scale": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: q=0 with arbitrary positive values
    input_dict = {"q": 0, "loc": 42, "scale": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: q=1 with negative location and larger scale
    input_dict = {"q": 1, "loc": -50, "scale": 25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.ppf_3"] = cauchy_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.ppf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.ppf_3'.")


check_valid('jax.scipy.stats.cauchy.ppf', generated_inputs['jax.scipy.stats.cauchy.ppf_3'], lib="jax", suffix=3)
