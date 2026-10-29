
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_not_inputs():
    list_of_inputs = []

    # Input 1: standard boolean True
    input_dict = {"x": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: standard boolean False
    input_dict = {"x": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: numpy boolean True
    input_dict = {"x": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: numpy boolean False
    input_dict = {"x": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: python bool from integer 1
    input_dict = {"x": bool(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: python bool from integer 0
    input_dict = {"x": bool(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy bool from integer 1
    input_dict = {"x": np.bool_(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy bool from integer 0
    input_dict = {"x": np.bool_(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: numpy True_
    input_dict = {"x": np.True_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: numpy False_
    input_dict = {"x": np.False_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.bitwise_not_3"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_not_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_not_3'.")


check_valid('jax.lax.bitwise_not', generated_inputs['jax.lax.bitwise_not_3'], lib="jax", suffix=3)
