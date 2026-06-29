
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nextafter_inputs():
    list_of_inputs = []

    # Input 1: Zero towards positive
    list_of_inputs.append({
        "x1": float(0.0),
        "x2": float(1.0)
    })

    # Input 2: Zero towards negative
    list_of_inputs.append({
        "x1": float(0.0),
        "x2": float(-1.0)
    })

    # Input 3: Positive float towards infinity
    list_of_inputs.append({
        "x1": float(1.0),
        "x2": float(2.0)
    })

    # Input 4: Positive float towards zero
    list_of_inputs.append({
        "x1": float(1.0),
        "x2": float(0.0)
    })

    # Input 5: Negative float towards negative infinity
    list_of_inputs.append({
        "x1": float(-1.0),
        "x2": float(-2.0)
    })

    # Input 6: Negative float towards zero
    list_of_inputs.append({
        "x1": float(-1.0),
        "x2": float(0.0)
    })

    # Input 7: Very small float towards positive
    list_of_inputs.append({
        "x1": float(1e-30),
        "x2": float(1.0)
    })

    # Input 8: Very large float towards zero
    list_of_inputs.append({
        "x1": float(1e30),
        "x2": float(0.0)
    })

    # Input 9: Infinity towards zero
    list_of_inputs.append({
        "x1": float('inf'),
        "x2": float(0.0)
    })

    # Input 10: Negative infinity towards zero
    list_of_inputs.append({
        "x1": float('-inf'),
        "x2": float(0.0)
    })

    # Input 11: Identical values
    list_of_inputs.append({
        "x1": float(1.5),
        "x2": float(1.5)
    })

    return list_of_inputs

generated_inputs["jax.lax.nextafter_2"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.nextafter_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.nextafter_2'.")


check_valid('jax.lax.nextafter', generated_inputs['jax.lax.nextafter_2'], lib="jax", suffix=2)
