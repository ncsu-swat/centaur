
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hilbert_inputs():
    list_of_inputs = []

    # Input 1: order 0
    input_dict = {"n": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: order 1
    input_dict = {"n": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: order 2
    input_dict = {"n": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: order 3
    input_dict = {"n": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: order 4
    input_dict = {"n": int(4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: order 5 using np.int32
    input_dict = {"n": int(np.int32(5))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: order 8 using np.int64
    input_dict = {"n": int(np.int64(8))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: order 10
    input_dict = {"n": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: order 16
    input_dict = {"n": 16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: order 32
    input_dict = {"n": 32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: order 64
    input_dict = {"n": 64}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
