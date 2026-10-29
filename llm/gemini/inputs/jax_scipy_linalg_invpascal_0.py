
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import sys
import numpy as np
import scipy.linalg
import jax.scipy.linalg

setattr(sys.modules["jax.scipy.linalg"], "invpascal", scipy.linalg.invpascal)


def invpascal_inputs():
    list_of_inputs = []

    # Input 1: size 1, symmetric
    input_dict = {"n": 1, "kind": "symmetric"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: size 2, lower triangular
    input_dict = {"n": 2, "kind": "lower"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: size 3, upper triangular
    input_dict = {"n": 3, "kind": "upper"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: size 4, symmetric
    input_dict = {"n": 4, "kind": "symmetric"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: size 5, lower triangular
    input_dict = {"n": 5, "kind": "lower"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: size 6, upper triangular
    input_dict = {"n": 6, "kind": "upper"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: size 7, symmetric
    input_dict = {"n": 7, "kind": "symmetric"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: size 8, lower triangular
    input_dict = {"n": 8, "kind": "lower"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: size 9, upper triangular
    input_dict = {"n": 9, "kind": "upper"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: size 10, symmetric
    input_dict = {"n": 10, "kind": "symmetric"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: size 12, lower triangular
    input_dict = {"n": 12, "kind": "lower"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: size 15, upper triangular
    input_dict = {"n": 15, "kind": "upper"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.invpascal"] = invpascal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.invpascal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.invpascal'.")


check_valid('jax.scipy.linalg.invpascal', generated_inputs['jax.scipy.linalg.invpascal'], lib="jax", suffix=0)
