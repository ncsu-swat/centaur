
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy


def heaviside_inputs():
    list_of_inputs = []

    # Input 1: x1 is zero, x2 is 0.5
    input_dict = {"x1": 0.0, "x2": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: x1 is negative, x2 is positive
    input_dict = {"x1": -3.5, "x2": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: x1 is positive, x2 is zero
    input_dict = {"x1": 4.2, "x2": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Both are zero
    input_dict = {"x1": 0.0, "x2": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x1 is zero, x2 is negative
    input_dict = {"x1": 0.0, "x2": -2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x1 is large positive, x2 is fractional
    input_dict = {"x1": 1e6, "x2": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x1 is small negative, x2 is large
    input_dict = {"x1": -1e-5, "x2": 100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x1 is negative zero, x2 is 0.75
    input_dict = {"x1": -0.0, "x2": 0.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x1 is positive, x2 is negative
    input_dict = {"x1": 12.8, "x2": -0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: x1 is negative, x2 is negative
    input_dict = {"x1": -100.0, "x2": -50.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: x1 is zero, x2 is large positive
    input_dict = {"x1": 0.0, "x2": 999.9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.heaviside_6"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_6'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_6'], lib="jax", suffix=6)
