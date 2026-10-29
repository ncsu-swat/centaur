
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isreal_inputs():
    list_of_inputs = []

    # Input 1: zero float
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive float
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative float
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive decimal float
    input_dict = {"x": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative decimal float
    input_dict = {"x": -2.718281828459045}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: very small positive float
    input_dict = {"x": 1e-10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: very large negative float
    input_dict = {"x": -1e10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: negative infinity
    input_dict = {"x": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: NaN float
    input_dict = {"x": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: extremely large float
    input_dict = {"x": 1e300}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: subnormal-scale small float
    input_dict = {"x": 1e-300}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isreal_3"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isreal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isreal_3'.")


check_valid('jax.numpy.isreal', generated_inputs['jax.numpy.isreal_3'], lib="jax", suffix=3)
