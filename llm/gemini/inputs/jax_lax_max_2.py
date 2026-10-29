
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def lax_max_inputs():
    list_of_inputs = []

    # Input 1: Zero values
    list_of_inputs.append({"x": 0, "y": 0})

    # Input 2: Both positive integers
    list_of_inputs.append({"x": 1, "y": 2})

    # Input 3: Negative and positive
    list_of_inputs.append({"x": -5, "y": 5})

    # Input 4: Both negative integers
    list_of_inputs.append({"x": -20, "y": -10})

    # Input 5: First argument strictly greater
    list_of_inputs.append({"x": 100, "y": 50})

    # Input 6: Negative and zero
    list_of_inputs.append({"x": -100, "y": 0})

    # Input 7: Identical non-zero values
    list_of_inputs.append({"x": 42, "y": 42})

    # Input 8: Close positive integers
    list_of_inputs.append({"x": 1000, "y": 999})

    # Input 9: Large negative values
    list_of_inputs.append({"x": -9999, "y": -10000})

    # Input 10: Positive and negative
    list_of_inputs.append({"x": 7, "y": -3})

    # Input 11: Larger range integers
    list_of_inputs.append({"x": 123456, "y": 654321})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.lax.max_2"] = lax_max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.max_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.max_2'.")


check_valid('jax.lax.max', generated_inputs['jax.lax.max_2'], lib="jax", suffix=2)
