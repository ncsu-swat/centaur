
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def lgamma_inputs():
    list_of_inputs = []

    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 6.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 7.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 8.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 9.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 20.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"x": 100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.lgamma_3"] = lgamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.lgamma_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.lgamma_3'.")


check_valid('jax.lax.lgamma', generated_inputs['jax.lax.lgamma_3'], lib="jax", suffix=3)
