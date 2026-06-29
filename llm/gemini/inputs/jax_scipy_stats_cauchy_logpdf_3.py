
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_cauchy_logpdf_inputs():
    list_of_inputs = []

    # Input 1: standard standard Cauchy
    list_of_inputs.append({
        "x": 0,
        "loc": 0,
        "scale": 1
    })

    # Input 2: positive x
    list_of_inputs.append({
        "x": 1,
        "loc": 0,
        "scale": 1
    })

    # Input 3: negative x
    list_of_inputs.append({
        "x": -1,
        "loc": 0,
        "scale": 1
    })

    # Input 4: shifted and scaled positive integers
    list_of_inputs.append({
        "x": 2,
        "loc": 1,
        "scale": 2
    })

    # Input 5: negative location and x, positive scale
    list_of_inputs.append({
        "x": -5,
        "loc": -2,
        "scale": 3
    })

    # Input 6: x matches loc
    list_of_inputs.append({
        "x": 10,
        "loc": 10,
        "scale": 5
    })

    # Input 7: x at 0, negative loc
    list_of_inputs.append({
        "x": 0,
        "loc": -10,
        "scale": 10
    })

    # Input 8: large negative x, large scale
    list_of_inputs.append({
        "x": -100,
        "loc": 0,
        "scale": 50
    })

    # Input 9: scale is 1, non-zero parameters
    list_of_inputs.append({
        "x": 5,
        "loc": 5,
        "scale": 1
    })

    # Input 10: negative x, positive loc, scale > 1
    list_of_inputs.append({
        "x": -3,
        "loc": 3,
        "scale": 4
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.logpdf_3"] = jax_scipy_stats_cauchy_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logpdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logpdf_3'.")


check_valid('jax.scipy.stats.cauchy.logpdf', generated_inputs['jax.scipy.stats.cauchy.logpdf_3'], lib="jax", suffix=3)
