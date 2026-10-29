
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cbrt_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive one
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative one
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive perfect cube
    input_dict = {"x": 8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative perfect cube
    input_dict = {"x": -8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger positive perfect cube
    input_dict = {"x": 27}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger negative perfect cube
    input_dict = {"x": -27}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Non-perfect positive cube
    input_dict = {"x": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-perfect negative cube
    input_dict = {"x": -50}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger positive integer
    input_dict = {"x": 1000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Larger negative integer
    input_dict = {"x": -1000000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.cbrt_3"] = cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cbrt_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cbrt_3'.")


check_valid('jax.numpy.cbrt', generated_inputs['jax.numpy.cbrt_3'], lib="jax", suffix=3)
