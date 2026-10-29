
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: zero x1 with standard 0.5 float
    input_dict = {"x1": 0, "x2": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative x1 with 0.5 float
    input_dict = {"x1": -1, "x2": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: positive x1 with 0.5 float
    input_dict = {"x1": 1, "x2": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: zero x1 with 1.0 float
    input_dict = {"x1": 0, "x2": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative x1 with zero float
    input_dict = {"x1": -5, "x2": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: positive x1 with negative float
    input_dict = {"x1": 10, "x2": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: zero x1 with negative float
    input_dict = {"x1": 0, "x2": -1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: zero x1 with fractional float
    input_dict = {"x1": 0, "x2": 0.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large negative x1 with positive float
    input_dict = {"x1": -100, "x2": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large positive x1 with zero float
    input_dict = {"x1": 42, "x2": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: zero x1 with infinity float
    input_dict = {"x1": 0, "x2": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: zero x1 with NaN float
    input_dict = {"x1": 0, "x2": float("nan")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.heaviside_8"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_8'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_8'], lib="jax", suffix=8)
