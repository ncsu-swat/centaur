
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def lax_max_inputs():
    list_of_inputs = []

    # Input 1: True, True
    list_of_inputs.append({"x": True, "y": True})

    # Input 2: True, False
    list_of_inputs.append({"x": True, "y": False})

    # Input 3: False, True
    list_of_inputs.append({"x": False, "y": True})

    # Input 4: False, False
    list_of_inputs.append({"x": False, "y": False})

    # Input 5: True, True
    list_of_inputs.append({"x": True, "y": True})

    # Input 6: True, False
    list_of_inputs.append({"x": True, "y": False})

    # Input 7: False, True
    list_of_inputs.append({"x": False, "y": True})

    # Input 8: False, False
    list_of_inputs.append({"x": False, "y": False})

    # Input 9: True, False
    list_of_inputs.append({"x": True, "y": False})

    # Input 10: False, True
    list_of_inputs.append({"x": False, "y": True})

    # Input 11: True, True
    list_of_inputs.append({"x": True, "y": True})

    # Input 12: False, False
    list_of_inputs.append({"x": False, "y": False})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.lax.max_4"] = lax_max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.max_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.max_4'.")


check_valid('jax.lax.max', generated_inputs['jax.lax.max_4'], lib="jax", suffix=4)
