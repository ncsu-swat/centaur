
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log1p_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: One
    input_dict = {"x": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative one (boundary value, log(0))
    input_dict = {"x": int(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive integer
    input_dict = {"x": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Moderate positive integer
    input_dict = {"x": int(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger positive integer
    input_dict = {"x": int(100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large positive integer
    input_dict = {"x": int(10000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative integer beyond -1 (returns nan in log)
    input_dict = {"x": int(-5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative integer
    input_dict = {"x": int(-100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Very large integer
    input_dict = {"x": int(1000000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Arbitrary integer
    input_dict = {"x": int(42)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.log1p_3"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log1p_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log1p_3'.")


check_valid('jax.numpy.log1p', generated_inputs['jax.numpy.log1p_3'], lib="jax", suffix=3)
