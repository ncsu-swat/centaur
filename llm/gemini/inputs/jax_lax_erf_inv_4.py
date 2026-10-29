
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def erf_inv_inputs():
    list_of_inputs = []

    # erf_inv accepts floating-point values in the domain (-1, 1)
    values = [
        0.0,
        0.5,
        -0.5,
        0.25,
        -0.25,
        0.75,
        -0.75,
        np.float32(0.1),
        np.float32(-0.1),
        np.float64(0.8),
        np.float64(-0.8),
        np.float32(0.6),
    ]

    for v in values:
        input_dict = {"x": v}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.erf_inv_4"] = erf_inv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erf_inv_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erf_inv_4'.")


check_valid('jax.lax.erf_inv', generated_inputs['jax.lax.erf_inv_4'], lib="jax", suffix=4)
