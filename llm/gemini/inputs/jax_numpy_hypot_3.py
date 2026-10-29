
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def hypot_inputs():
    list_of_inputs = []

    # Input 1: Classic Pythagorean triple (3, 4)
    input_dict = {"x1": 3, "x2": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero values
    input_dict = {"x1": 0, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integers
    input_dict = {"x1": -3, "x2": -4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Pythagorean triple (5, 12)
    input_dict = {"x1": 5, "x2": 12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mixed signs with Pythagorean triple (-8, 15)
    input_dict = {"x1": -8, "x2": 15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: One zero and one positive integer
    input_dict = {"x1": 0, "x2": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: One positive integer and one zero
    input_dict = {"x1": 7, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Mixed signs with larger values (20, -21)
    input_dict = {"x1": 20, "x2": -21}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive integers
    input_dict = {"x1": 1000, "x2": 2000}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative and positive large integers
    input_dict = {"x1": -9999, "x2": 1234}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Unit values
    input_dict = {"x1": 1, "x2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Negative integer and zero
    input_dict = {"x1": -100, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.hypot_3"] = hypot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hypot_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hypot_3'.")


check_valid('jax.numpy.hypot', generated_inputs['jax.numpy.hypot_3'], lib="jax", suffix=3)
