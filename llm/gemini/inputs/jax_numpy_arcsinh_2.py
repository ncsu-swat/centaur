
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def arcsinh_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive unit value
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative unit value
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive fractional value
    input_dict = {"x": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small negative fractional value
    input_dict = {"x": -0.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Moderate positive value
    input_dict = {"x": 5.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Moderate negative value
    input_dict = {"x": -12.3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large positive value
    input_dict = {"x": 1000.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large negative value
    input_dict = {"x": -1000.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Very small positive number close to zero
    input_dict = {"x": 1e-6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Very small negative number close to zero
    input_dict = {"x": -1e-6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Positive infinity
    input_dict = {"x": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.arcsinh_2"] = arcsinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arcsinh_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arcsinh_2'.")


check_valid('jax.numpy.arcsinh', generated_inputs['jax.numpy.arcsinh_2'], lib="jax", suffix=2)
