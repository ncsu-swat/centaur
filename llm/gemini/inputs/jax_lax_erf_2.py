
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def erf_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive standard value
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative standard value
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small positive value
    input_dict = {"x": 0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small negative value
    input_dict = {"x": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive value
    input_dict = {"x": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large negative value
    input_dict = {"x": -5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small positive value close to zero
    input_dict = {"x": 1e-7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Positive fraction / pi
    input_dict = {"x": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative float
    input_dict = {"x": -2.718281828459045}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Very large magnitude positive float
    input_dict = {"x": 100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative zero
    input_dict = {"x": -0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.erf_2"] = erf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erf_2'.")


check_valid('jax.lax.erf', generated_inputs['jax.lax.erf_2'], lib="jax", suffix=2)
