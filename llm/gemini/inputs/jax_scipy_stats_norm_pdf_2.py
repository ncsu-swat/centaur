
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_pdf_inputs():
    list_of_inputs = []

    # Input 1: Standard normal at 0
    input_dict = {
        "x": float(0.0),
        "loc": float(0.0),
        "scale": float(1.0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive x, positive mean
    input_dict = {
        "x": float(1.5),
        "loc": float(1.0),
        "scale": float(0.5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x, negative mean
    input_dict = {
        "x": float(-2.5),
        "loc": float(-2.0),
        "scale": float(1.2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large scale
    input_dict = {
        "x": float(10.0),
        "loc": float(0.0),
        "scale": float(100.0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small scale
    input_dict = {
        "x": float(0.05),
        "loc": float(0.0),
        "scale": float(0.01),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x matches loc
    input_dict = {
        "x": float(3.14159),
        "loc": float(3.14159),
        "scale": float(2.0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative loc with positive x
    input_dict = {
        "x": float(4.0),
        "loc": float(-3.0),
        "scale": float(2.5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Positive loc with negative x
    input_dict = {
        "x": float(-5.0),
        "loc": float(2.0),
        "scale": float(1.5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive value
    input_dict = {
        "x": float(1e3),
        "loc": float(1e3),
        "scale": float(10.0),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Fractional values
    input_dict = {
        "x": float(0.333),
        "loc": float(0.125),
        "scale": float(0.875),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Non-integer scale with small offset
    input_dict = {
        "x": float(-0.7071),
        "loc": float(0.0),
        "scale": float(1.4142),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.norm.pdf_2"] = norm_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.pdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.pdf_2'.")


check_valid('jax.scipy.stats.norm.pdf', generated_inputs['jax.scipy.stats.norm.pdf_2'], lib="jax", suffix=2)
