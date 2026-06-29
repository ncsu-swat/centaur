
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def helmert_inputs():
    list_of_inputs = []

    # Input 1: size 1, full matrix
    list_of_inputs.append({"n": 1, "full": True})

    # Input 2: size 2, contrast block
    list_of_inputs.append({"n": 2, "full": False})

    # Input 3: size 2, full matrix
    list_of_inputs.append({"n": 2, "full": True})

    # Input 4: size 3, contrast block
    list_of_inputs.append({"n": 3, "full": False})

    # Input 5: size 3, full matrix
    list_of_inputs.append({"n": 3, "full": True})

    # Input 6: size 5, contrast block
    list_of_inputs.append({"n": 5, "full": False})

    # Input 7: size 5, full matrix
    list_of_inputs.append({"n": 5, "full": True})

    # Input 8: size 10, contrast block
    list_of_inputs.append({"n": 10, "full": False})

    # Input 9: size 10, full matrix
    list_of_inputs.append({"n": 10, "full": True})

    # Input 10: size 50, contrast block
    list_of_inputs.append({"n": 50, "full": False})

    # Input 11: size 100, full matrix
    list_of_inputs.append({"n": 100, "full": True})

    return list_of_inputs

generated_inputs["jax.scipy.linalg.helmert"] = helmert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.helmert' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.helmert'.")


check_valid('jax.scipy.linalg.helmert', generated_inputs['jax.scipy.linalg.helmert'], lib="jax", suffix=0)
