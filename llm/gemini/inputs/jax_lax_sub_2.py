
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def sub_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integers
    input_dict = {"x": 10, "y": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Both zeros
    input_dict = {"x": 0, "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative and positive integer
    input_dict = {"x": -3, "y": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive and negative integer
    input_dict = {"x": 15, "y": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Both negative integers
    input_dict = {"x": -20, "y": -8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Equal positive integers
    input_dict = {"x": 42, "y": 42}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Subtraction resulting in negative value
    input_dict = {"x": 3, "y": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger values
    input_dict = {"x": 12345, "y": 6789}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero and negative integer
    input_dict = {"x": 0, "y": -100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Equal negative integers
    input_dict = {"x": -1, "y": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Powers of two
    input_dict = {"x": 1024, "y": 512}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.sub_2"] = sub_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sub_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sub_2'.")


check_valid('jax.lax.sub', generated_inputs['jax.lax.sub_2'], lib="jax", suffix=2)
