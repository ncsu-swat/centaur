
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def lax_max_inputs():
    list_of_inputs = []

    # Input 1: Zero values
    list_of_inputs.append({"x": 0.0, "y": 0.0})

    # Input 2: Positive floats
    list_of_inputs.append({"x": 1.5, "y": 2.5})

    # Input 3: Negative floats
    list_of_inputs.append({"x": -5.5, "y": -2.3})

    # Input 4: Mixed positive and negative floats
    list_of_inputs.append({"x": 100.0, "y": -100.0})

    # Input 5: Float constants
    list_of_inputs.append({"x": 3.141592653589793, "y": 2.718281828459045})

    # Input 6: Positive infinity
    list_of_inputs.append({"x": float("inf"), "y": 1e30})

    # Input 7: Negative infinity
    list_of_inputs.append({"x": float("-inf"), "y": -1e30})

    # Input 8: Very small positive floats
    list_of_inputs.append({"x": 1e-15, "y": 1e-14})

    # Input 9: Signed zeroes
    list_of_inputs.append({"x": -0.0, "y": 0.0})

    # Input 10: Equal values
    list_of_inputs.append({"x": 42.0, "y": 42.0})

    # Input 11: Very large numbers
    list_of_inputs.append({"x": 1e100, "y": 2e100})

    # Input 12: Close negative floats
    list_of_inputs.append({"x": -1e-5, "y": -1e-4})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.lax.max_3"] = lax_max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.max_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.max_3'.")


check_valid('jax.lax.max', generated_inputs['jax.lax.max_3'], lib="jax", suffix=3)
