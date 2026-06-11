
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import os

os.environ["JAX_PLATFORMS"] = "cpu"

def jax_lax_erf_inputs():
    list_of_inputs = []
    
    values = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 0.1]
    for val in values:
        list_of_inputs.append({"x": np.float32(val)})
        
    return list_of_inputs

generated_inputs["jax.lax.erf_2"] = jax_lax_erf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erf_2'.")


check_valid('jax.lax.erf', generated_inputs['jax.lax.erf_2'], lib="jax", suffix=2)
