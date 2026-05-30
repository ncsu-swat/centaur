
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ndim_inputs():
    list_of_inputs = []

    # Input 1: standard positive float
    input_dict = {"a": 3.14}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: standard negative float
    input_dict = {"a": -2.718}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero float
    input_dict = {"a": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: large float
    input_dict = {"a": 1.5e15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: small negative float
    input_dict = {"a": -1.2e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: positive infinity
    input_dict = {"a": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: negative infinity
    input_dict = {"a": float('-inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Not a Number (NaN)
    input_dict = {"a": float('nan')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: extremely small float
    input_dict = {"a": 1e-30}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: precise decimal float
    input_dict = {"a": 12345.6789}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ndim_3"] = ndim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ndim_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ndim_3'.")


check_valid('jax.numpy.ndim', generated_inputs['jax.numpy.ndim_3'], lib="jax", suffix=3)
