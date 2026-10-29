
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sqrt_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    input_dict = {"x": 4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative zero
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Non-integer float
    input_dict = {"x": 2.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small positive float
    input_dict = {"x": 1e-12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive float
    input_dict = {"x": 1e12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Fractional float less than 1
    input_dict = {"x": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative float (produces nan)
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large negative float (produces nan)
    input_dict = {"x": -42.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative infinity
    input_dict = {"x": float("-inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: NaN
    input_dict = {"x": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.sqrt_3"] = sqrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sqrt_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sqrt_3'.")


check_valid('jax.numpy.sqrt', generated_inputs['jax.numpy.sqrt_3'], lib="jax", suffix=3)
