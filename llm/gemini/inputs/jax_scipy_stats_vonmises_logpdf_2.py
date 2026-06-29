
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_vonmises_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Basic positive values
    input_dict = {"x": float(0.0), "kappa": float(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Non-zero mean, larger kappa
    input_dict = {"x": float(1.0), "kappa": float(2.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x value, small kappa
    input_dict = {"x": float(-1.0), "kappa": float(0.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Near pi, large kappa
    input_dict = {"x": float(3.14159), "kappa": float(10.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Near negative pi, very small kappa
    input_dict = {"x": float(-3.14159), "kappa": float(0.1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero kappa (uniform distribution)
    input_dict = {"x": float(0.5), "kappa": float(0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: High concentration (high kappa)
    input_dict = {"x": float(2.5), "kappa": float(50.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative x, moderate kappa
    input_dict = {"x": float(-2.0), "kappa": float(5.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small positive x, very small kappa
    input_dict = {"x": float(1.5), "kappa": float(0.01)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative fraction, standard kappa
    input_dict = {"x": float(-0.75), "kappa": float(1.25)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Out of standard [-pi, pi] interval (should wrap due to periodicity)
    input_dict = {"x": float(10.0), "kappa": float(2.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.vonmises.logpdf_2"] = jax_scipy_stats_vonmises_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.vonmises.logpdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.vonmises.logpdf_2'.")


check_valid('jax.scipy.stats.vonmises.logpdf', generated_inputs['jax.scipy.stats.vonmises.logpdf_2'], lib="jax", suffix=2)
