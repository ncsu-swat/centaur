
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iterable_inputs():
    list_of_inputs = []

    # Input 1: positive integer
    list_of_inputs.append({"y": 5})

    # Input 2: negative integer
    list_of_inputs.append({"y": -10})

    # Input 3: zero
    list_of_inputs.append({"y": 0})

    # Input 4: large positive integer
    list_of_inputs.append({"y": 123456789})

    # Input 5: small positive integer
    list_of_inputs.append({"y": 1})

    # Input 6: small negative integer
    list_of_inputs.append({"y": -1})

    # Input 7: another positive integer
    list_of_inputs.append({"y": 100})

    # Input 8: another negative integer
    list_of_inputs.append({"y": -100})

    # Input 9: larger positive integer
    list_of_inputs.append({"y": 999999})

    # Input 10: larger negative integer
    list_of_inputs.append({"y": -999999})

    # Input 11: positive integer
    list_of_inputs.append({"y": 2})

    # Input 12: negative integer
    list_of_inputs.append({"y": -2})

    return list_of_inputs

generated_inputs["jax.numpy.iterable_4"] = iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_4'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_4'], lib="jax", suffix=4)
