
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def nextafter_inputs():
    list_of_inputs = []

    # Input 1: positive direction from zero
    input_dict = {"x1": 0.0, "x2": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive direction from 1.0
    input_dict = {"x1": 1.0, "x2": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative direction towards zero
    input_dict = {"x1": 1.0, "x2": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: negative values, decreasing
    input_dict = {"x1": -1.0, "x2": -2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative direction from zero
    input_dict = {"x1": 0.0, "x2": -1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: identical values
    input_dict = {"x1": 2.5, "x2": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: towards positive infinity
    input_dict = {"x1": 1e30, "x2": float("inf")}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: from positive infinity towards zero
    input_dict = {"x1": float("inf"), "x2": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: from negative infinity towards zero
    input_dict = {"x1": float("-inf"), "x2": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: small scale positive values
    input_dict = {"x1": 1e-10, "x2": 1e-9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: negative to positive direction
    input_dict = {"x1": -5.0, "x2": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.nextafter_2"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.nextafter_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.nextafter_2'.")


check_valid('jax.lax.nextafter', generated_inputs['jax.lax.nextafter_2'], lib="jax", suffix=2)
