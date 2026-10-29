
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def xlogy_inputs():
    list_of_inputs = []

    # Input 1: (0.0, 0.0) special defined point
    input_dict = {"x": 0.0, "y": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: standard positive floats
    input_dict = {"x": 1.0, "y": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: y equals e (Euler's number)
    input_dict = {"x": 2.5, "y": float(np.e)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: negative x, positive y
    input_dict = {"x": -3.0, "y": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x is zero, positive y
    input_dict = {"x": 0.0, "y": 5.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: small fractional floats
    input_dict = {"x": 0.5, "y": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: large positive values
    input_dict = {"x": 1e4, "y": 1e6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: small positive values near zero
    input_dict = {"x": 1e-6, "y": 1e-3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: negative x with larger y
    input_dict = {"x": -0.75, "y": 100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: positive x, y less than 1
    input_dict = {"x": 10.0, "y": 0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: x is zero, negative y (returns 0.0 by definition)
    input_dict = {"x": 0.0, "y": -4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: non-integer float values
    input_dict = {"x": 3.14159, "y": 2.71828}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.special.xlogy_2"] = xlogy_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.xlogy_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.xlogy_2'.")


check_valid('jax.scipy.special.xlogy', generated_inputs['jax.scipy.special.xlogy_2'], lib="jax", suffix=2)
