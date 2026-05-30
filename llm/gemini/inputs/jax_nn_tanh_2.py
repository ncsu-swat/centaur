
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_nn_tanh_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard negative float
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero float
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive float
    input_dict = {"x": 50.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative float
    input_dict = {"x": -50.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small positive float
    input_dict = {"x": 1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small negative float
    input_dict = {"x": -1e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Floating point representation of e
    input_dict = {"x": 2.71828}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Floating point representation of -e
    input_dict = {"x": -2.71828}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float value between 0 and 1
    input_dict = {"x": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Float value between -1 and 0
    input_dict = {"x": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.tanh_2"] = jax_nn_tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.tanh_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.tanh_2'.")


check_valid('jax.nn.tanh', generated_inputs['jax.nn.tanh_2'], lib="jax", suffix=2)
