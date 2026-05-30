
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def triu_indices_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix, main diagonal
    input_dict = {
        "n": int(3),
        "k": int(0),
        "m": int(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Square matrix, one diagonal above main
    input_dict = {
        "n": int(4),
        "k": int(1),
        "m": int(4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Square matrix, one diagonal below main
    input_dict = {
        "n": int(4),
        "k": int(-1),
        "m": int(4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Rectangular matrix (wide), main diagonal
    input_dict = {
        "n": int(3),
        "k": int(0),
        "m": int(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular matrix (tall), main diagonal
    input_dict = {
        "n": int(5),
        "k": int(0),
        "m": int(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rectangular matrix (wide), positive k
    input_dict = {
        "n": int(3),
        "k": int(2),
        "m": int(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rectangular matrix (tall), negative k
    input_dict = {
        "n": int(5),
        "k": int(-2),
        "m": int(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Minimal size 1x1
    input_dict = {
        "n": int(1),
        "k": int(0),
        "m": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger dimensions
    input_dict = {
        "n": int(10),
        "k": int(3),
        "m": int(12)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger dimensions with large negative k
    input_dict = {
        "n": int(15),
        "k": int(-5),
        "m": int(15)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
