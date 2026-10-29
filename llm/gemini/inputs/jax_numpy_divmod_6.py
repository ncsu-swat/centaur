
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def divmod_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integers
    input_dict = {"x1": 10, "x2": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Evenly divisible integers
    input_dict = {"x1": 20, "x2": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative dividend
    input_dict = {"x1": -15, "x2": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative divisor
    input_dict = {"x1": 15, "x2": -4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Both negative
    input_dict = {"x1": -25, "x2": -6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero dividend
    input_dict = {"x1": 0, "x2": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Dividend smaller than divisor
    input_dict = {"x1": 3, "x2": 8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger integers
    input_dict = {"x1": 1000000, "x2": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Divisor is 1
    input_dict = {"x1": -42, "x2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Divisor is -1
    input_dict = {"x1": 50, "x2": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Powers of two
    input_dict = {"x1": 1024, "x2": 64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.divmod_6"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_6'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_6'], lib="jax", suffix=6)
