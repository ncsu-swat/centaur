
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expm1_inputs():
    list_of_inputs = []

    # Input 1: positive float
    input_dict = {"x": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative float
    input_dict = {"x": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: small positive float (close to zero)
    input_dict = {"x": 1e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: small negative float (close to zero)
    input_dict = {"x": -1e-4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: larger positive float
    input_dict = {"x": 5.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: larger negative float
    input_dict = {"x": -10.2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: very small positive float
    input_dict = {"x": 1e-12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: very small negative float
    input_dict = {"x": -1e-12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float with pi value
    input_dict = {"x": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.expm1_2"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expm1_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expm1_2'.")


check_valid('jax.numpy.expm1', generated_inputs['jax.numpy.expm1_2'], lib="jax", suffix=2)
