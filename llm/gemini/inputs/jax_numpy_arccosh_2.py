
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arccosh_inputs():
    list_of_inputs = []

    # Input 1: Boundary value at x = 1.0 (arccosh(1.0) = 0.0)
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small value greater than 1
    input_dict = {"x": 1.0001}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Standard float value
    input_dict = {"x": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Moderate float value
    input_dict = {"x": 5.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Value yielding a known integer-ish result
    input_dict = {"x": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large float value
    input_dict = {"x": 1e6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very large float value
    input_dict = {"x": 1e15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Value less than 1 (returns nan)
    input_dict = {"x": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero (returns nan)
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative float value (returns nan)
    input_dict = {"x": -3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: NaN input
    input_dict = {"x": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.arccosh_2"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccosh_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccosh_2'.")


check_valid('jax.numpy.arccosh', generated_inputs['jax.numpy.arccosh_2'], lib="jax", suffix=2)
