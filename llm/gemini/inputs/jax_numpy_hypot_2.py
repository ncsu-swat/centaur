
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hypot_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats
    input_dict = {"x1": 3.0, "x2": 4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative floats
    input_dict = {"x1": -3.0, "x2": -4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed signs
    input_dict = {"x1": -5.0, "x2": 12.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zeros
    input_dict = {"x1": 0.0, "x2": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: One zero, one positive
    input_dict = {"x1": 0.0, "x2": 5.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large floats
    input_dict = {"x1": 1e150, "x2": 1e150}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very small floats
    input_dict = {"x1": 1e-150, "x2": 1e-150}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Infinity
    input_dict = {"x1": float('inf'), "x2": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: NaN value
    input_dict = {"x1": float('nan'), "x2": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Decimals
    input_dict = {"x1": 1.234, "x2": 5.678}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative infinity
    input_dict = {"x1": float('-inf'), "x2": float('-inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.hypot_2"] = hypot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hypot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hypot_2'.")


check_valid('jax.numpy.hypot', generated_inputs['jax.numpy.hypot_2'], lib="jax", suffix=2)
