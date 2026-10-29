
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arctanh_inputs():
    list_of_inputs = []

    # Input 1: zero
    input_dict = {"x": float(0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: standard positive float within domain
    input_dict = {"x": float(0.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: standard negative float within domain
    input_dict = {"x": float(-0.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: close to positive boundary
    input_dict = {"x": float(0.999)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: close to negative boundary
    input_dict = {"x": float(-0.999)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: exact positive boundary
    input_dict = {"x": float(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: exact negative boundary
    input_dict = {"x": float(-1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: small positive float
    input_dict = {"x": float(1e-6)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: small negative float
    input_dict = {"x": float(-1e-6)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float value outside [-1, 1] (returns nan)
    input_dict = {"x": float(2.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: negative float value outside [-1, 1]
    input_dict = {"x": float(-3.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.arctanh_2"] = arctanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arctanh_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arctanh_2'.")


check_valid('jax.numpy.arctanh', generated_inputs['jax.numpy.arctanh_2'], lib="jax", suffix=2)
