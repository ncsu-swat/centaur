
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_gennorm_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Zero mean, beta=1 (Laplace distribution equivalent)
    list_of_inputs.append({"x": 0.0, "beta": 1.0})

    # Input 2: Standard positive value, beta=2 (Normal distribution equivalent)
    list_of_inputs.append({"x": 1.0, "beta": 2.0})

    # Input 3: Negative value, beta=2
    list_of_inputs.append({"x": -1.0, "beta": 2.0})

    # Input 4: Small positive value, fractional beta < 1
    list_of_inputs.append({"x": 0.5, "beta": 0.5})

    # Input 5: Negative value, moderate beta
    list_of_inputs.append({"x": -2.5, "beta": 1.5})

    # Input 6: Large positive value, beta=3
    list_of_inputs.append({"x": 10.0, "beta": 3.0})

    # Input 7: Large negative value, beta=3
    list_of_inputs.append({"x": -10.0, "beta": 3.0})

    # Input 8: Small positive value, very small beta
    list_of_inputs.append({"x": 0.001, "beta": 0.1})

    # Input 9: Moderate value, large beta
    list_of_inputs.append({"x": 5.5, "beta": 10.0})

    # Input 10: Negative value, beta > 1
    list_of_inputs.append({"x": -0.5, "beta": 1.2})

    return list_of_inputs

generated_inputs["jax.scipy.stats.gennorm.logpdf_3"] = jax_scipy_stats_gennorm_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.logpdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.logpdf_3'.")


check_valid('jax.scipy.stats.gennorm.logpdf', generated_inputs['jax.scipy.stats.gennorm.logpdf_3'], lib="jax", suffix=3)
