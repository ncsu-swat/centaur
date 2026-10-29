
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def fmod_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integers
    input_dict = {"x1": 5, "x2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative dividend, positive divisor
    input_dict = {"x1": -5, "x2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Positive dividend, negative divisor
    input_dict = {"x1": 5, "x2": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Both negative
    input_dict = {"x1": -5, "x2": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Dividend is zero
    input_dict = {"x1": 0, "x2": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Dividend smaller than divisor
    input_dict = {"x1": 3, "x2": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Exact division (no remainder)
    input_dict = {"x1": 42, "x2": 6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Dividend equals divisor
    input_dict = {"x1": 17, "x2": 17}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive integers
    input_dict = {"x1": 1000003, "x2": 97}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large negative dividend
    input_dict = {"x1": -987654, "x2": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Powers of two
    input_dict = {"x1": 1024, "x2": 32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Divisor is 1
    input_dict = {"x1": 1234, "x2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fmod_7"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_7'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_7'], lib="jax", suffix=7)
