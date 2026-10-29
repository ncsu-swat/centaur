
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bernoulli_inputs():
    list_of_inputs = []

    # Input 1: minimal positive integer
    input_dict = {"n": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: small integer
    input_dict = {"n": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: integer 3
    input_dict = {"n": int(3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: integer 4
    input_dict = {"n": int(4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: integer 5
    input_dict = {"n": int(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: integer 6
    input_dict = {"n": int(6)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: integer 8
    input_dict = {"n": int(8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: integer 10
    input_dict = {"n": int(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: integer 15
    input_dict = {"n": int(15)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: integer 20
    input_dict = {"n": int(20)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: integer 30
    input_dict = {"n": int(30)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.special.bernoulli"] = bernoulli_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.bernoulli' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.bernoulli'.")


check_valid('jax.scipy.special.bernoulli', generated_inputs['jax.scipy.special.bernoulli'], lib="jax", suffix=0)
