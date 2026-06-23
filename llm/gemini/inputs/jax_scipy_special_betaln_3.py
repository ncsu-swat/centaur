
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import copy

def betaln_inputs():
    list_of_inputs = []
    for i in range(1, 11):
        list_of_inputs.append({
            "a": np.int32(i),
            "b": np.int32(i + 1)
        })
    return list_of_inputs

generated_inputs["jax.scipy.special.betaln_3"] = betaln_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.betaln_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.betaln_3'.")


check_valid('jax.scipy.special.betaln', generated_inputs['jax.scipy.special.betaln_3'], lib="jax", suffix=3)
