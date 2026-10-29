
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logistic_pdf_inputs():
    list_of_inputs = []

    # Input 1: Standard logistic distribution at x=0
    input_dict = {"x": int(0), "loc": int(0), "scale": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive x, standard loc and scale
    input_dict = {"x": int(1), "loc": int(0), "scale": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x, standard loc and scale
    input_dict = {"x": int(-1), "loc": int(0), "scale": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive loc and scale > 1
    input_dict = {"x": int(5), "loc": int(2), "scale": int(3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative loc and negative x
    input_dict = {"x": int(-5), "loc": int(-2), "scale": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x equals loc
    input_dict = {"x": int(10), "loc": int(10), "scale": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x at 0 with positive loc
    input_dict = {"x": int(0), "loc": int(5), "scale": int(4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative x and positive loc
    input_dict = {"x": int(-10), "loc": int(10), "scale": int(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer values
    input_dict = {"x": int(100), "loc": int(0), "scale": int(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Positive x and negative loc with prime scale
    input_dict = {"x": int(42), "loc": int(-42), "scale": int(7)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative x, positive loc, scale of 2
    input_dict = {"x": int(-3), "loc": int(3), "scale": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.logistic.pdf_3"] = logistic_pdf_inputs()

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
