
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def floor_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    input_dict = {"x": 3.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard negative float
    input_dict = {"x": -3.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative zero
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive float
    input_dict = {"x": 123456.789}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large negative float
    input_dict = {"x": -987654.321}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small positive float close to zero
    input_dict = {"x": 1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small negative float close to zero
    input_dict = {"x": -1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Integer-valued float
    input_dict = {"x": 42.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative integer-valued float
    input_dict = {"x": -100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Positive infinity
    input_dict = {"x": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative infinity
    input_dict = {"x": float('-inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.floor_3"] = floor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_3'.")


check_valid('jax.numpy.floor', generated_inputs['jax.numpy.floor_3'], lib="jax", suffix=3)
