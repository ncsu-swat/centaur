
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_gumbel_r_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Standard Gumbel distribution at mode
    list_of_inputs.append({
        'x': float(0.0),
        'loc': float(0.0),
        'scale': float(1.0)
    })

    # Input 2: Right tail
    list_of_inputs.append({
        'x': float(2.5),
        'loc': float(0.0),
        'scale': float(1.0)
    })

    # Input 3: Left tail
    list_of_inputs.append({
        'x': float(-2.5),
        'loc': float(0.0),
        'scale': float(1.0)
    })

    # Input 4: Shifted distribution with small scale
    list_of_inputs.append({
        'x': float(1.5),
        'loc': float(1.0),
        'scale': float(0.2)
    })

    # Input 5: Large positive parameters
    list_of_inputs.append({
        'x': float(10.0),
        'loc': float(5.0),
        'scale': float(2.5)
    })

    # Input 6: Negative location parameter
    list_of_inputs.append({
        'x': float(-5.0),
        'loc': float(-10.0),
        'scale': float(3.0)
    })

    # Input 7: Very small scale parameter
    list_of_inputs.append({
        'x': float(0.1),
        'loc': float(0.0),
        'scale': float(0.05)
    })

    # Input 8: Large scale parameter
    list_of_inputs.append({
        'x': float(50.0),
        'loc': float(10.0),
        'scale': float(100.0)
    })

    # Input 9: Large negative x
    list_of_inputs.append({
        'x': float(-50.0),
        'loc': float(0.0),
        'scale': float(10.0)
    })

    # Input 10: Equal inputs
    list_of_inputs.append({
        'x': float(0.5),
        'loc': float(0.5),
        'scale': float(0.5)
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_r.logcdf_2"] = jax_scipy_stats_gumbel_r_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.logcdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.logcdf_2'.")


check_valid('jax.scipy.stats.gumbel_r.logcdf', generated_inputs['jax.scipy.stats.gumbel_r.logcdf_2'], lib="jax", suffix=2)
