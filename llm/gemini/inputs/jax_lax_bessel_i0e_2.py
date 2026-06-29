
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_bessel_i0e_inputs():
    list_of_inputs = []

    # Input 1: positive python float
    list_of_inputs.append({"x": 1.5})

    # Input 2: negative python float
    list_of_inputs.append({"x": -2.5})

    # Input 3: zero python float
    list_of_inputs.append({"x": 0.0})

    # Input 4: large positive python float
    list_of_inputs.append({"x": 100.0})

    # Input 5: large negative python float
    list_of_inputs.append({"x": -100.0})

    # Input 6: positive np.float32 scalar
    list_of_inputs.append({"x": np.float32(0.5)})

    # Input 7: negative np.float32 scalar
    list_of_inputs.append({"x": np.float32(-3.2)})

    # Input 8: positive np.float64 scalar
    list_of_inputs.append({"x": np.float64(10.25)})

    # Input 9: negative np.float64 scalar
    list_of_inputs.append({"x": np.float64(-15.75)})

    # Input 10: small positive python float
    list_of_inputs.append({"x": 1e-5})

    # Input 11: zero np.float64 scalar
    list_of_inputs.append({"x": np.float64(0.0)})

    return list_of_inputs

generated_inputs["jax.lax.bessel_i0e_2"] = jax_lax_bessel_i0e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i0e_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i0e_2'.")


check_valid('jax.lax.bessel_i0e', generated_inputs['jax.lax.bessel_i0e_2'], lib="jax", suffix=2)
