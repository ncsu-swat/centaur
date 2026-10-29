
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bernoulli_ppf_inputs():
    list_of_inputs = []

    # Input 1: standard int 0, 0
    input_dict = {"q": 0, "p": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: standard int 0, 1
    input_dict = {"q": 0, "p": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: standard int 1, 0
    input_dict = {"q": 1, "p": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: standard int 1, 1
    input_dict = {"q": 1, "p": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: numpy int32
    input_dict = {"q": np.int32(0), "p": np.int32(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy int64
    input_dict = {"q": np.int64(1), "p": np.int64(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: negative integer for q
    input_dict = {"q": -1, "p": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: integer > 1 for q
    input_dict = {"q": 2, "p": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: negative integer for p
    input_dict = {"q": 0, "p": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: numpy int16 with values > 1
    input_dict = {"q": np.int16(5), "p": np.int16(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: numpy int8
    input_dict = {"q": np.int8(1), "p": np.int8(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: both negative
    input_dict = {"q": -3, "p": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.bernoulli.ppf_3"] = bernoulli_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.bernoulli.ppf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.bernoulli.ppf_3'.")


check_valid('jax.scipy.stats.bernoulli.ppf', generated_inputs['jax.scipy.stats.bernoulli.ppf_3'], lib="jax", suffix=3)
