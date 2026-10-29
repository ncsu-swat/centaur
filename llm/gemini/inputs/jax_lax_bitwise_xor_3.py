
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import jax.lax


def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: (True, False) using Python booleans
    list_of_inputs.append({"x": True, "y": False})

    # Input 2: (False, True) using Python booleans
    list_of_inputs.append({"x": False, "y": True})

    # Input 3: (True, True) using Python booleans
    list_of_inputs.append({"x": True, "y": True})

    # Input 4: (False, False) using Python booleans
    list_of_inputs.append({"x": False, "y": False})

    # Input 5: (True, False) using numpy boolean scalars
    list_of_inputs.append({"x": np.bool_(True), "y": np.bool_(False)})

    # Input 6: (False, True) using numpy boolean scalars
    list_of_inputs.append({"x": np.bool_(False), "y": np.bool_(True)})

    # Input 7: (True, True) using numpy boolean scalars
    list_of_inputs.append({"x": np.bool_(True), "y": np.bool_(True)})

    # Input 8: (False, False) using numpy boolean scalars
    list_of_inputs.append({"x": np.bool_(False), "y": np.bool_(False)})

    # Input 9: Mixed Python bool and numpy bool (True, False)
    list_of_inputs.append({"x": True, "y": np.bool_(False)})

    # Input 10: Mixed numpy bool and Python bool (True, False)
    list_of_inputs.append({"x": np.bool_(True), "y": False})

    # Input 11: Mixed Python bool and numpy bool (False, True)
    list_of_inputs.append({"x": False, "y": np.bool_(True)})

    # Input 12: Mixed numpy bool and Python bool (False, True)
    list_of_inputs.append({"x": np.bool_(False), "y": True})

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.lax.bitwise_xor_3"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_xor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_xor_3'.")


check_valid('jax.lax.bitwise_xor', generated_inputs['jax.lax.bitwise_xor_3'], lib="jax", suffix=3)
