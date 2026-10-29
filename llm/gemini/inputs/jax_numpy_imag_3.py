
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def imag_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"val": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive standard float
    input_dict = {"val": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float
    input_dict = {"val": -1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive float
    input_dict = {"val": 1e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small negative float
    input_dict = {"val": -1e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large float
    input_dict = {"val": 1e12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Pi float
    input_dict = {"val": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative large float
    input_dict = {"val": -1e12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Positive infinity
    input_dict = {"val": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative infinity
    input_dict = {"val": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: NaN
    input_dict = {"val": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.imag_3"] = imag_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.imag_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.imag_3'.")


check_valid('jax.numpy.imag', generated_inputs['jax.numpy.imag_3'], lib="jax", suffix=3)
