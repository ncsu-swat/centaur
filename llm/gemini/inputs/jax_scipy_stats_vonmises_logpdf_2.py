
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vonmises_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Basic positive values within [-pi, pi]
    input_dict = {"x": float(0.0), "kappa": float(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Non-zero positive x, kappa zero
    input_dict = {"x": float(np.pi / 2), "kappa": float(0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x within [-pi, pi], moderate kappa
    input_dict = {"x": float(-np.pi / 4), "kappa": float(2.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boundary values of the support [-pi, pi]
    input_dict = {"x": float(np.pi), "kappa": float(0.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Lower boundary -pi
    input_dict = {"x": float(-np.pi), "kappa": float(3.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large kappa (concentrated distribution)
    input_dict = {"x": float(0.1), "kappa": float(50.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very small positive kappa
    input_dict = {"x": float(-1.2), "kappa": float(1e-4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large kappa
    input_dict = {"x": float(0.0), "kappa": float(100.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x outside [-pi, pi] (vonmises is 2*pi periodic)
    input_dict = {"x": float(3 * np.pi), "kappa": float(1.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative x outside [-pi, pi]
    input_dict = {"x": float(-2.5 * np.pi), "kappa": float(4.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Arbitrary float values
    input_dict = {"x": float(1.2345), "kappa": float(0.8765)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.vonmises.logpdf_2"] = vonmises_logpdf_inputs()

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
