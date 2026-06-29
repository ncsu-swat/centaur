
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_scipy_stats_logistic_pdf_inputs():
    list_of_inputs = []

    # Input 1: Basic standard logistic
    input_dict = {"x": 0, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive x, standard scale
    input_dict = {"x": 2, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x, standard scale
    input_dict = {"x": -2, "loc": 0, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Shifted loc
    input_dict = {"x": 3, "loc": 3, "scale": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large scale
    input_dict = {"x": 5, "loc": 0, "scale": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative loc, scale > 1
    input_dict = {"x": -1, "loc": -5, "scale": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All positive integers
    input_dict = {"x": 10, "loc": 5, "scale": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large scale, x at loc
    input_dict = {"x": 10, "loc": 10, "scale": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative x and loc
    input_dict = {"x": -10, "loc": -10, "scale": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Custom integer parameters
    input_dict = {"x": 12, "loc": -4, "scale": 8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.logistic.pdf_3"] = (
    jax_scipy_stats_logistic_pdf_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.pdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.pdf_3'.")


check_valid('jax.scipy.stats.logistic.pdf', generated_inputs['jax.scipy.stats.logistic.pdf_3'], lib="jax", suffix=3)
