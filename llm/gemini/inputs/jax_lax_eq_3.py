
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def lax_eq_inputs():
    list_of_inputs = []

    # Input 1: Equal positive floats
    list_of_inputs.append({"x": 1.0, "y": 1.0})

    # Input 2: Unequal positive floats
    list_of_inputs.append({"x": 2.5, "y": 3.75})

    # Input 3: Negative and positive float
    list_of_inputs.append({"x": -1.5, "y": 1.5})

    # Input 4: Equal negative floats
    list_of_inputs.append({"x": -42.0, "y": -42.0})

    # Input 5: Positive zero and negative zero
    list_of_inputs.append({"x": 0.0, "y": -0.0})

    # Input 6: Very large floats
    list_of_inputs.append({"x": 1e20, "y": 1e20})

    # Input 7: Very small floats
    list_of_inputs.append({"x": 1e-15, "y": 2e-15})

    # Input 8: Positive infinity
    list_of_inputs.append({"x": float("inf"), "y": float("inf")})

    # Input 9: Negative infinity vs finite float
    list_of_inputs.append({"x": float("-inf"), "y": -1e30})

    # Input 10: NaN comparisons
    list_of_inputs.append({"x": float("nan"), "y": float("nan")})

    # Input 11: Zero vs small positive float
    list_of_inputs.append({"x": 0.0, "y": 1e-7})

    # Input 12: High-precision differing floats
    list_of_inputs.append({"x": 3.141592653589793, "y": 3.141592653589794})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.lax.eq_3"] = lax_eq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.eq_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.eq_3'.")


check_valid('jax.lax.eq', generated_inputs['jax.lax.eq_3'], lib="jax", suffix=3)
