
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def triu_indices_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix, main diagonal
    list_of_inputs.append({"n": 3, "k": 0, "m": 3})

    # Input 2: Square matrix, first super-diagonal
    list_of_inputs.append({"n": 4, "k": 1, "m": 4})

    # Input 3: Square matrix, first sub-diagonal
    list_of_inputs.append({"n": 4, "k": -1, "m": 4})

    # Input 4: Rectangular matrix where m > n, main diagonal
    list_of_inputs.append({"n": 3, "k": 0, "m": 5})

    # Input 5: Rectangular matrix where n > m, main diagonal
    list_of_inputs.append({"n": 5, "k": 0, "m": 2})

    # Input 6: Rectangular matrix, positive offset k
    list_of_inputs.append({"n": 6, "k": 2, "m": 4})

    # Input 7: Rectangular matrix, negative offset k
    list_of_inputs.append({"n": 5, "k": -3, "m": 6})

    # Input 8: Minimal 1x1 matrix
    list_of_inputs.append({"n": 1, "k": 0, "m": 1})

    # Input 9: Large offset exceeding matrix dimensions (empty result)
    list_of_inputs.append({"n": 3, "k": 10, "m": 3})

    # Input 10: Negative offset covering the entire matrix
    list_of_inputs.append({"n": 4, "k": -5, "m": 4})

    # Input 11: Larger matrix with moderate offset
    list_of_inputs.append({"n": 10, "k": 3, "m": 12})

    # Input 12: Empty matrix dimensions
    list_of_inputs.append({"n": 0, "k": 0, "m": 0})

    return list_of_inputs


generated_inputs["jax.numpy.triu_indices"] = triu_indices_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.triu_indices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.triu_indices'.")


check_valid('jax.numpy.triu_indices', generated_inputs['jax.numpy.triu_indices'], lib="jax", suffix=0)
