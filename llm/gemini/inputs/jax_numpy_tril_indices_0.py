
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tril_indices_inputs():
    list_of_inputs = []

    # Input 1: Square matrix, diagonal k=0
    input_dict = {"n": 3, "k": 0, "m": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Square matrix, positive k
    input_dict = {"n": 3, "k": 1, "m": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Square matrix, negative k
    input_dict = {"n": 3, "k": -1, "m": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rectangular matrix (n < m), k=0
    input_dict = {"n": 3, "k": 0, "m": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular matrix (n > m), k=0
    input_dict = {"n": 5, "k": 0, "m": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rectangular matrix, positive k
    input_dict = {"n": 4, "k": 2, "m": 6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rectangular matrix, negative k
    input_dict = {"n": 6, "k": -2, "m": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger dimensions
    input_dict = {"n": 10, "k": 0, "m": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive k exceeding dimensions
    input_dict = {"n": 4, "k": 10, "m": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large negative k
    input_dict = {"n": 5, "k": -5, "m": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Single element matrix
    input_dict = {"n": 1, "k": 0, "m": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.tril_indices"] = tril_indices_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tril_indices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tril_indices'.")


check_valid('jax.numpy.tril_indices', generated_inputs['jax.numpy.tril_indices'], lib="jax", suffix=0)
