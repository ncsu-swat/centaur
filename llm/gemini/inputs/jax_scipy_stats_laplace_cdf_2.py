
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_laplace_cdf_inputs():
    list_of_inputs = []

    # Input 1: Standard Laplace, zero mean, scale 1
    list_of_inputs.append({
        'x': float(0.0),
        'loc': float(0.0),
        'scale': float(1.0)
    })

    # Input 2: Positive values, larger scale
    list_of_inputs.append({
        'x': float(1.5),
        'loc': float(0.0),
        'scale': float(2.0)
    })

    # Input 3: Negative x, positive loc, small scale
    list_of_inputs.append({
        'x': float(-1.0),
        'loc': float(1.0),
        'scale': float(0.5)
    })

    # Input 4: Large positive values
    list_of_inputs.append({
        'x': float(10.0),
        'loc': float(-5.0),
        'scale': float(3.5)
    })

    # Input 5: Negative x, negative loc, very small scale
    list_of_inputs.append({
        'x': float(-2.5),
        'loc': float(-1.0),
        'scale': float(0.1)
    })

    # Input 6: Zero x and loc, large scale
    list_of_inputs.append({
        'x': float(0.0),
        'loc': float(0.0),
        'scale': float(10.0)
    })

    # Input 7: Using numpy float64 types, large values
    list_of_inputs.append({
        'x': np.float64(100.0),
        'loc': np.float64(50.0),
        'scale': np.float64(20.0)
    })

    # Input 8: Using numpy float32 types
    list_of_inputs.append({
        'x': np.float32(-0.5),
        'loc': np.float32(0.5),
        'scale': np.float32(1.5)
    })

    # Input 9: Arbitrary fractional values
    list_of_inputs.append({
        'x': float(1.23),
        'loc': float(4.56),
        'scale': float(7.89)
    })

    # Input 10: Far left tail
    list_of_inputs.append({
        'x': float(-10.0),
        'loc': float(0.0),
        'scale': float(1.0)
    })

    # Input 11: Median point (x == loc)
    list_of_inputs.append({
        'x': float(5.0),
        'loc': float(5.0),
        'scale': float(1.0)
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.laplace.cdf_2"] = jax_scipy_stats_laplace_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.laplace.cdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.laplace.cdf_2'.")


check_valid('jax.scipy.stats.laplace.cdf', generated_inputs['jax.scipy.stats.laplace.cdf_2'], lib="jax", suffix=2)
