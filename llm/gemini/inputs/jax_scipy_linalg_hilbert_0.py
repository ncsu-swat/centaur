
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hilbert_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({"n": 1})

    # Input 2
    list_of_inputs.append({"n": 2})

    # Input 3
    list_of_inputs.append({"n": 3})

    # Input 4
    list_of_inputs.append({"n": 4})

    # Input 5
    list_of_inputs.append({"n": 5})

    # Input 6
    list_of_inputs.append({"n": 8})

    # Input 7
    list_of_inputs.append({"n": 10})

    # Input 8
    list_of_inputs.append({"n": 16})

    # Input 9
    list_of_inputs.append({"n": 32})

    # Input 10
    list_of_inputs.append({"n": 64})

    # Input 11
    list_of_inputs.append({"n": 128})

    return list_of_inputs

generated_inputs["jax.scipy.linalg.hilbert"] = hilbert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.hilbert' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.hilbert'.")


check_valid('jax.scipy.linalg.hilbert', generated_inputs['jax.scipy.linalg.hilbert'], lib="jax", suffix=0)
