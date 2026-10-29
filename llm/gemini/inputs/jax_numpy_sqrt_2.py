
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def sqrt_inputs():
    list_of_inputs = []

    # Input 1: zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: one
    input_dict = {"x": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: small perfect square
    input_dict = {"x": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: non-perfect square positive integer
    input_dict = {"x": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: another perfect square
    input_dict = {"x": 9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: negative one (returns nan)
    input_dict = {"x": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: larger negative integer
    input_dict = {"x": -16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: medium perfect square
    input_dict = {"x": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: prime integer
    input_dict = {"x": 17}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large positive integer
    input_dict = {"x": 1000000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: negative prime integer
    input_dict = {"x": -7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.sqrt_2"] = sqrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sqrt_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sqrt_2'.")


check_valid('jax.numpy.sqrt', generated_inputs['jax.numpy.sqrt_2'], lib="jax", suffix=2)
