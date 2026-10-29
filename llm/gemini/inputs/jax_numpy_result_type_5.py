
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def result_type_inputs():
    list_of_inputs = []

    # Input 1: zero float
    input_dict = {"args": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive unit float
    input_dict = {"args": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative float
    input_dict = {"args": -1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive floating point constant (pi)
    input_dict = {"args": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative floating point constant (e)
    input_dict = {"args": -2.718281828459045}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: small positive magnitude float
    input_dict = {"args": 1e-12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large positive magnitude float
    input_dict = {"args": 1e15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: negative large float
    input_dict = {"args": -1e20}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: positive infinity
    input_dict = {"args": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: negative infinity
    input_dict = {"args": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: NaN value
    input_dict = {"args": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: negative zero
    input_dict = {"args": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.result_type_5"] = result_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.result_type_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.result_type_5'.")


check_valid('jax.numpy.result_type', generated_inputs['jax.numpy.result_type_5'], lib="jax", suffix=5)
