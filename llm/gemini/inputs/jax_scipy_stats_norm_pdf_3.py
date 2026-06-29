
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_norm_pdf_inputs():
    list_of_inputs = []

    # Input 1: standard normal mean 0, scale 1, evaluate at 0
    list_of_inputs.append({"x": int(0), "loc": int(0), "scale": int(1)})
    
    # Input 2: evaluate at 1 with standard normal
    list_of_inputs.append({"x": int(1), "loc": int(0), "scale": int(1)})
    
    # Input 3: negative x value, scale 2
    list_of_inputs.append({"x": int(-1), "loc": int(0), "scale": int(2)})
    
    # Input 4: positive mean and larger scale
    list_of_inputs.append({"x": int(5), "loc": int(2), "scale": int(3)})
    
    # Input 5: negative mean, negative x, scale 4
    list_of_inputs.append({"x": int(-10), "loc": int(-5), "scale": int(4)})
    
    # Input 6: large x value, scale 10
    list_of_inputs.append({"x": int(100), "loc": int(0), "scale": int(10)})
    
    # Input 7: x equal to mean, scale 1
    list_of_inputs.append({"x": int(2), "loc": int(2), "scale": int(1)})
    
    # Input 8: negative x, positive mean, scale 5
    list_of_inputs.append({"x": int(-3), "loc": int(3), "scale": int(5)})
    
    # Input 9: x at 0, negative mean, scale 20
    list_of_inputs.append({"x": int(0), "loc": int(-10), "scale": int(20)})
    
    # Input 10: large positive mean, x equal to mean, scale 1
    list_of_inputs.append({"x": int(42), "loc": int(42), "scale": int(1)})

    return list_of_inputs

generated_inputs["jax.scipy.stats.norm.pdf_3"] = jax_scipy_stats_norm_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.pdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.pdf_3'.")


check_valid('jax.scipy.stats.norm.pdf', generated_inputs['jax.scipy.stats.norm.pdf_3'], lib="jax", suffix=3)
