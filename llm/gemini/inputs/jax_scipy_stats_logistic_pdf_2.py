
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_logistic_pdf_inputs():
    list_of_inputs = []

    # Input 1: Standard logistic distribution PDF at 0
    list_of_inputs.append({
        'x': 0.0,
        'loc': 0.0,
        'scale': 1.0
    })

    # Input 2: Positive x, standard loc and scale
    list_of_inputs.append({
        'x': 1.5,
        'loc': 0.0,
        'scale': 1.0
    })

    # Input 3: Negative x, standard loc and scale
    list_of_inputs.append({
        'x': -1.5,
        'loc': 0.0,
        'scale': 1.0
    })

    # Input 4: Standard x and scale, shifted loc
    list_of_inputs.append({
        'x': 0.0,
        'loc': 2.5,
        'scale': 1.0
    })

    # Input 5: x equals loc, standard scale
    list_of_inputs.append({
        'x': 2.5,
        'loc': 2.5,
        'scale': 1.0
    })

    # Input 6: Standard x and loc, larger scale
    list_of_inputs.append({
        'x': 0.0,
        'loc': 0.0,
        'scale': 2.0
    })

    # Input 7: x equals loc, smaller scale
    list_of_inputs.append({
        'x': 1.0,
        'loc': 1.0,
        'scale': 0.5
    })

    # Input 8: Negative x and loc, larger scale
    list_of_inputs.append({
        'x': -5.0,
        'loc': -2.0,
        'scale': 3.5
    })

    # Input 9: Positive x and loc, very small scale
    list_of_inputs.append({
        'x': 10.0,
        'loc': 5.0,
        'scale': 0.1
    })

    # Input 10: Negative x, positive loc, scale > 1
    list_of_inputs.append({
        'x': -0.5,
        'loc': 0.5,
        'scale': 1.5
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.pdf_2"] = jax_scipy_stats_logistic_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.pdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.pdf_2'.")


check_valid('jax.scipy.stats.logistic.pdf', generated_inputs['jax.scipy.stats.logistic.pdf_2'], lib="jax", suffix=2)
