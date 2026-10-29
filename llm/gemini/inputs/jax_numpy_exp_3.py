
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp
import numpy as np


def exp_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive small integer
    input_dict = {"x": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative small integer
    input_dict = {"x": int(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Moderate positive integer
    input_dict = {"x": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Moderate negative integer
    input_dict = {"x": int(-5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Positive integer 5
    input_dict = {"x": int(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Positive integer 10
    input_dict = {"x": int(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative integer -10
    input_dict = {"x": int(-10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative integer -20
    input_dict = {"x": int(-20)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger positive integer within single-precision range
    input_dict = {"x": int(50)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Larger negative integer
    input_dict = {"x": int(-50)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.exp_3"] = exp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.exp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.exp_3'.")


check_valid('jax.numpy.exp', generated_inputs['jax.numpy.exp_3'], lib="jax", suffix=3)
