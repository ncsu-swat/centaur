
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def gennorm_logpdf_inputs():
    list_of_inputs = []

    # Input 1: standard normal case (beta = 2.0)
    input_dict = {"x": 0.0, "beta": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Laplace-like distribution (beta = 1.0) with negative x
    input_dict = {"x": -1.5, "beta": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Sub-Gaussian / heavy-tailed (beta < 1)
    input_dict = {"x": 2.5, "beta": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small negative x with beta > 2
    input_dict = {"x": -0.01, "beta": 3.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large positive x with beta = 4.0
    input_dict = {"x": 10.0, "beta": 4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Moderate negative x with fractional beta
    input_dict = {"x": -5.2, "beta": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Very small beta near 0
    input_dict = {"x": 0.0, "beta": 0.1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Positive x with large beta approaching uniform
    input_dict = {"x": 1.23, "beta": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large negative x
    input_dict = {"x": -25.0, "beta": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-integer x and beta
    input_dict = {"x": 3.14159, "beta": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative x with beta < 1
    input_dict = {"x": -2.718, "beta": 0.8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: High shape parameter (beta = 10.0)
    input_dict = {"x": 0.5, "beta": 10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gennorm.logpdf_3"] = gennorm_logpdf_inputs()

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
