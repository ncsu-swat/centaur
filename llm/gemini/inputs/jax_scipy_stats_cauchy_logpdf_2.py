
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_cauchy_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Standard parameters
    list_of_inputs.append({
        "x": 0.0,
        "loc": 0.0,
        "scale": 1.0
    })

    # Input 2: Positive x, standard loc and scale
    list_of_inputs.append({
        "x": 1.5,
        "loc": 0.0,
        "scale": 1.0
    })

    # Input 3: Negative x, positive loc, larger scale
    list_of_inputs.append({
        "x": -2.5,
        "loc": 1.0,
        "scale": 2.0
    })

    # Input 4: Positive x, negative loc, small scale
    list_of_inputs.append({
        "x": 0.5,
        "loc": -1.0,
        "scale": 0.5
    })

    # Input 5: Large positive x, large positive loc, moderate scale
    list_of_inputs.append({
        "x": 10.0,
        "loc": 5.0,
        "scale": 3.5
    })

    # Input 6: Large negative x, negative loc, very small scale
    list_of_inputs.append({
        "x": -10.0,
        "loc": -5.0,
        "scale": 0.1
    })

    # Input 7: Zero x and loc, very large scale
    list_of_inputs.append({
        "x": 0.0,
        "loc": 0.0,
        "scale": 100.0
    })

    # Input 8: Floating values with decimals
    list_of_inputs.append({
        "x": 1.234,
        "loc": 5.678,
        "scale": 0.912
    })

    # Input 9: Small values near zero, scale 1.0
    list_of_inputs.append({
        "x": -0.001,
        "loc": 0.001,
        "scale": 1.0
    })

    # Input 10: Extreme distance, large scale
    list_of_inputs.append({
        "x": 100.0,
        "loc": 0.0,
        "scale": 50.0
    })

    # Input 11: Negative x, positive loc, medium scale
    list_of_inputs.append({
        "x": -50.0,
        "loc": 50.0,
        "scale": 10.0
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.logpdf_2"] = jax_scipy_stats_cauchy_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logpdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logpdf_2'.")


check_valid('jax.scipy.stats.cauchy.logpdf', generated_inputs['jax.scipy.stats.cauchy.logpdf_2'], lib="jax", suffix=2)
