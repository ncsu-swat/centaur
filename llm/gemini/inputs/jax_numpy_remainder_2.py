
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.numpy as jnp

def remainder_inputs():
    list_of_inputs = []

    # Input 1: positive dividend, positive divisor
    list_of_inputs.append({"x1": 10, "x2": 3})

    # Input 2: negative dividend, positive divisor
    list_of_inputs.append({"x1": -10, "x2": 3})

    # Input 3: positive dividend, negative divisor
    list_of_inputs.append({"x1": 10, "x2": -3})

    # Input 4: negative dividend, negative divisor
    list_of_inputs.append({"x1": -10, "x2": -3})

    # Input 5: zero dividend
    list_of_inputs.append({"x1": 0, "x2": 5})

    # Input 6: exact multiple
    list_of_inputs.append({"x1": 15, "x2": 5})

    # Input 7: dividend smaller than divisor
    list_of_inputs.append({"x1": 3, "x2": 7})

    # Input 8: negative dividend smaller in magnitude than divisor
    list_of_inputs.append({"x1": -1, "x2": 2})

    # Input 9: larger integers
    list_of_inputs.append({"x1": 1000003, "x2": 97})

    # Input 10: divisor of 1
    list_of_inputs.append({"x1": 42, "x2": 1})

    # Input 11: divisor of -1
    list_of_inputs.append({"x1": 42, "x2": -1})

    # Input 12: negative multiple
    list_of_inputs.append({"x1": -20, "x2": 4})

    return list_of_inputs

generated_inputs["jax.numpy.remainder_2"] = remainder_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.remainder_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.remainder_2'.")


check_valid('jax.numpy.remainder', generated_inputs['jax.numpy.remainder_2'], lib="jax", suffix=2)
