
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bessel_i1e_inputs():
    list_of_inputs = []

    # Input 1: positive float
    list_of_inputs.append({"x": 1.0})

    # Input 2: negative float
    list_of_inputs.append({"x": -1.0})

    # Input 3: zero
    list_of_inputs.append({"x": 0.0})

    # Input 4: small positive float
    list_of_inputs.append({"x": 1e-5})

    # Input 5: small negative float
    list_of_inputs.append({"x": -1e-5})

    # Input 6: large positive float
    list_of_inputs.append({"x": 100.0})

    # Input 7: large negative float
    list_of_inputs.append({"x": -100.0})

    # Input 8: float value 0.5
    list_of_inputs.append({"x": 0.5})

    # Input 9: float value pi
    list_of_inputs.append({"x": 3.141592653589793})

    # Input 10: float value e
    list_of_inputs.append({"x": 2.718281828459045})

    return list_of_inputs

generated_inputs["jax.lax.bessel_i1e_2"] = bessel_i1e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i1e_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i1e_2'.")


check_valid('jax.lax.bessel_i1e', generated_inputs['jax.lax.bessel_i1e_2'], lib="jax", suffix=2)
