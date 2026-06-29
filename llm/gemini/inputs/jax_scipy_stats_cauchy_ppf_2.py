
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_cauchy_ppf_inputs():
    list_of_inputs = []

    # Input 1: Standard median
    input_dict = {
        "q": float(0.5),
        "loc": float(0.0),
        "scale": float(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Lower quartile, negative loc, larger scale
    input_dict = {
        "q": float(0.25),
        "loc": float(-2.5),
        "scale": float(1.5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Upper quartile, positive loc, smaller scale
    input_dict = {
        "q": float(0.75),
        "loc": float(3.0),
        "scale": float(0.5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Low probability, small scale
    input_dict = {
        "q": float(0.05),
        "loc": float(0.0),
        "scale": float(0.1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: High probability, large scale
    input_dict = {
        "q": float(0.95),
        "loc": float(10.0),
        "scale": float(10.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Near zero probability, large negative loc
    input_dict = {
        "q": float(0.001),
        "loc": float(-100.0),
        "scale": float(5.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Near one probability, large positive loc
    input_dict = {
        "q": float(0.999),
        "loc": float(100.0),
        "scale": float(5.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small scale, zero loc
    input_dict = {
        "q": float(0.4),
        "loc": float(0.0),
        "scale": float(0.01)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large scale, negative loc
    input_dict = {
        "q": float(0.6),
        "loc": float(-50.0),
        "scale": float(100.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Standard Cauchy with extreme q
    input_dict = {
        "q": float(0.99),
        "loc": float(0.0),
        "scale": float(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.ppf_2"] = jax_scipy_stats_cauchy_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.ppf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.ppf_2'.")


check_valid('jax.scipy.stats.cauchy.ppf', generated_inputs['jax.scipy.stats.cauchy.ppf_2'], lib="jax", suffix=2)
