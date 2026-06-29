
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_norm_ppf_inputs():
    list_of_inputs = []

    # Input 1: Standard normal median
    list_of_inputs.append({
        "q": float(0.5),
        "loc": float(0.0),
        "scale": float(1.0)
    })

    # Input 2: Lower tail
    list_of_inputs.append({
        "q": float(0.05),
        "loc": float(0.0),
        "scale": float(1.0)
    })

    # Input 3: Upper tail
    list_of_inputs.append({
        "q": float(0.95),
        "loc": float(0.0),
        "scale": float(1.0)
    })

    # Input 4: Shifted and scaled distribution
    list_of_inputs.append({
        "q": float(0.5),
        "loc": float(5.0),
        "scale": float(2.5)
    })

    # Input 5: Shifted and scaled, lower tail
    list_of_inputs.append({
        "q": float(0.16),
        "loc": float(-3.0),
        "scale": float(0.5)
    })

    # Input 6: Float32 numpy scalar
    list_of_inputs.append({
        "q": np.float32(0.75),
        "loc": np.float32(1.0),
        "scale": np.float32(2.0)
    })

    # Input 7: Float64 numpy scalar
    list_of_inputs.append({
        "q": np.float64(0.25),
        "loc": np.float64(-10.0),
        "scale": np.float64(4.0)
    })

    # Input 8: Extreme upper tail
    list_of_inputs.append({
        "q": float(0.999),
        "loc": float(0.0),
        "scale": float(1.0)
    })

    # Input 9: Extreme lower tail
    list_of_inputs.append({
        "q": float(0.001),
        "loc": float(0.0),
        "scale": float(1.0)
    })

    # Input 10: Small scale parameter
    list_of_inputs.append({
        "q": float(0.5),
        "loc": float(0.0),
        "scale": float(1e-5)
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.norm.ppf_2"] = jax_scipy_stats_norm_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.ppf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.ppf_2'.")


check_valid('jax.scipy.stats.norm.ppf', generated_inputs['jax.scipy.stats.norm.ppf_2'], lib="jax", suffix=2)
