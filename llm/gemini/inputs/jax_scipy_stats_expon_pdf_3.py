
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expon_pdf_inputs():
    list_of_inputs = []

    # Input 1: Standard exponential
    list_of_inputs.append({
        'x': float(1.0),
        'loc': float(0.0),
        'scale': float(1.0)
    })

    # Input 2: Positive scale, shifted loc
    list_of_inputs.append({
        'x': float(2.5),
        'loc': float(1.0),
        'scale': float(2.0)
    })

    # Input 3: Small scale
    list_of_inputs.append({
        'x': float(0.5),
        'loc': float(0.0),
        'scale': float(0.5)
    })

    # Input 4: x less than loc
    list_of_inputs.append({
        'x': float(-1.0),
        'loc': float(0.0),
        'scale': float(1.0)
    })

    # Input 5: Negative loc, positive scale
    list_of_inputs.append({
        'x': float(5.0),
        'loc': float(-2.0),
        'scale': float(3.5)
    })

    # Input 6: Large values
    list_of_inputs.append({
        'x': float(10.0),
        'loc': float(5.0),
        'scale': float(10.0)
    })

    # Input 7: x at boundary loc
    list_of_inputs.append({
        'x': float(0.0),
        'loc': float(0.0),
        'scale': float(1.0)
    })

    # Input 8: Small scale, precise values
    list_of_inputs.append({
        'x': float(1.23),
        'loc': float(0.5),
        'scale': float(0.1)
    })

    # Input 9: Large x, large loc
    list_of_inputs.append({
        'x': float(100.0),
        'loc': float(50.0),
        'scale': float(2.5)
    })

    # Input 10: Negative x but x > loc
    list_of_inputs.append({
        'x': float(-5.5),
        'loc': float(-10.0),
        'scale': float(4.2)
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.expon.pdf_3"] = expon_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.pdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.pdf_3'.")


check_valid('jax.scipy.stats.expon.pdf', generated_inputs['jax.scipy.stats.expon.pdf_3'], lib="jax", suffix=3)
