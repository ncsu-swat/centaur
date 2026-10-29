
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def heaviside_inputs():
    list_of_inputs = []

    # Input 1: x1 is zero, x2 is positive
    list_of_inputs.append(copy.deepcopy({"x1": 0, "x2": 1}))

    # Input 2: x1 is negative, x2 is zero
    list_of_inputs.append(copy.deepcopy({"x1": -5, "x2": 0}))

    # Input 3: x1 is positive, x2 is positive
    list_of_inputs.append(copy.deepcopy({"x1": 10, "x2": 2}))

    # Input 4: x1 is zero, x2 is negative
    list_of_inputs.append(copy.deepcopy({"x1": 0, "x2": -1}))

    # Input 5: x1 is negative, x2 is large positive
    list_of_inputs.append(copy.deepcopy({"x1": -1, "x2": 100}))

    # Input 6: x1 is positive, x2 is zero
    list_of_inputs.append(copy.deepcopy({"x1": 1, "x2": 0}))

    # Input 7: x1 is zero, x2 is zero
    list_of_inputs.append(copy.deepcopy({"x1": 0, "x2": 0}))

    # Input 8: x1 is positive, x2 is negative
    list_of_inputs.append(copy.deepcopy({"x1": 42, "x2": -42}))

    # Input 9: x1 is negative, x2 is negative
    list_of_inputs.append(copy.deepcopy({"x1": -100, "x2": -50}))

    # Input 10: x1 is zero, x2 is large positive
    list_of_inputs.append(copy.deepcopy({"x1": 0, "x2": 999}))

    # Input 11: x1 is positive, x2 is positive
    list_of_inputs.append(copy.deepcopy({"x1": 7, "x2": 3}))

    return list_of_inputs


generated_inputs["jax.numpy.heaviside_9"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_9' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_9'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_9'], lib="jax", suffix=9)
