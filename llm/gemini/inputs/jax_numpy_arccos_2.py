
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arccos_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive boundary
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative boundary
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive mid-value
    input_dict = {"x": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative mid-value
    input_dict = {"x": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Approx pi/4 cos value
    input_dict = {"x": 0.70710678}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Approx negative pi/4 cos value
    input_dict = {"x": -0.70710678}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small positive value
    input_dict = {"x": 0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small negative value
    input_dict = {"x": -0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Value close to 1
    input_dict = {"x": 0.999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Value close to -1
    input_dict = {"x": -0.999}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.arccos_2"] = arccos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccos_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccos_2'.")


check_valid('jax.numpy.arccos', generated_inputs['jax.numpy.arccos_2'], lib="jax", suffix=2)
