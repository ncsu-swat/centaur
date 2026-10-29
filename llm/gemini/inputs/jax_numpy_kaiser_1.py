
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def kaiser_inputs():
    list_of_inputs = []

    # Input 1: Basic small window with float beta
    list_of_inputs.append(copy.deepcopy({"M": 4, "beta": 1.5}))

    # Input 2: Zero length window
    list_of_inputs.append(copy.deepcopy({"M": 0, "beta": 14.0}))

    # Input 3: Length 1 window with zero beta
    list_of_inputs.append(copy.deepcopy({"M": 1, "beta": 0.0}))

    # Input 4: Length 2 window
    list_of_inputs.append(copy.deepcopy({"M": 2, "beta": 2.0}))

    # Input 5: Odd window size with zero beta (boxcar window)
    list_of_inputs.append(copy.deepcopy({"M": 5, "beta": 0.0}))

    # Input 6: Typical signal processing window size with moderate beta
    list_of_inputs.append(copy.deepcopy({"M": 10, "beta": 5.0}))

    # Input 7: Window size 12 with standard Kaiser beta
    list_of_inputs.append(copy.deepcopy({"M": 12, "beta": 8.6}))

    # Input 8: Larger window with high beta
    list_of_inputs.append(copy.deepcopy({"M": 64, "beta": 14.0}))

    # Input 9: Large window size with fractional beta
    list_of_inputs.append(copy.deepcopy({"M": 128, "beta": 0.5}))

    # Input 10: Negative beta value
    list_of_inputs.append(copy.deepcopy({"M": 15, "beta": -3.5}))

    # Input 11: Even window size with large beta
    list_of_inputs.append(copy.deepcopy({"M": 256, "beta": 20.0}))

    # Input 12: Odd window size with small beta
    list_of_inputs.append(copy.deepcopy({"M": 33, "beta": 0.1}))

    return list_of_inputs


generated_inputs["jax.numpy.kaiser_1"] = kaiser_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.kaiser_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.kaiser_1'.")


check_valid('jax.numpy.kaiser', generated_inputs['jax.numpy.kaiser_1'], lib="jax", suffix=1)
