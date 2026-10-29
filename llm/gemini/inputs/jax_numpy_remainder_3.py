
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def remainder_inputs():
    list_of_inputs = []

    # Input 1: Positive dividend and positive divisor
    input_dict = {"x1": 10.0, "x2": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive dividend and negative divisor
    input_dict = {"x1": 10.0, "x2": -3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative dividend and positive divisor
    input_dict = {"x1": -10.0, "x2": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative dividend and negative divisor
    input_dict = {"x1": -10.0, "x2": -3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Non-integer dividend and divisor with exact division
    input_dict = {"x1": 7.5, "x2": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Dividend smaller than divisor
    input_dict = {"x1": 1.25, "x2": 4.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Fractional values
    input_dict = {"x1": 0.55, "x2": 0.12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero dividend
    input_dict = {"x1": 0.0, "x2": 5.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large magnitude values
    input_dict = {"x1": 1000000.75, "x2": 1234.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Small magnitude values
    input_dict = {"x1": 1e-4, "x2": 3e-5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Irrational approximations
    input_dict = {"x1": 3.1415926535, "x2": 2.7182818284}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.remainder_3"] = remainder_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.remainder_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.remainder_3'.")


check_valid('jax.numpy.remainder', generated_inputs['jax.numpy.remainder_3'], lib="jax", suffix=3)
