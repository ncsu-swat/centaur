
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polygamma_inputs():
    list_of_inputs = []

    # Input 1: Digamma (n = 0) at x = 1.0
    list_of_inputs.append({"n": int(0), "x": float(1.0)})

    # Input 2: Trigamma (n = 1) at x = 2.5
    list_of_inputs.append({"n": int(1), "x": float(2.5)})

    # Input 3: Digamma (n = 0) at fractional positive value
    list_of_inputs.append({"n": int(0), "x": float(0.5)})

    # Input 4: Tetragamma (n = 2) at x = 3.7
    list_of_inputs.append({"n": int(2), "x": float(3.7)})

    # Input 5: Trigamma (n = 1) at negative non-integer value
    list_of_inputs.append({"n": int(1), "x": float(-0.5)})

    # Input 6: Pentagamma (n = 3) at x = 1.2
    list_of_inputs.append({"n": int(3), "x": float(1.2)})

    # Input 7: Digamma (n = 0) at negative non-integer value
    list_of_inputs.append({"n": int(0), "x": float(-1.5)})

    # Input 8: Higher-order derivative (n = 4) at large positive x
    list_of_inputs.append({"n": int(4), "x": float(10.0)})

    # Input 9: Tetragamma (n = 2) at small positive x
    list_of_inputs.append({"n": int(2), "x": float(0.01)})

    # Input 10: Higher-order derivative (n = 5) at x = 2.0
    list_of_inputs.append({"n": int(5), "x": float(2.0)})

    # Input 11: Trigamma (n = 1) at large positive x
    list_of_inputs.append({"n": int(1), "x": float(100.0)})

    # Input 12: Digamma (n = 0) at x = 5.5
    list_of_inputs.append({"n": int(0), "x": float(5.5)})

    return list_of_inputs


generated_inputs["jax.scipy.special.polygamma_3"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.polygamma_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.polygamma_3'.")


check_valid('jax.scipy.special.polygamma', generated_inputs['jax.scipy.special.polygamma_3'], lib="jax", suffix=3)
