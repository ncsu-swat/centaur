
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def blackman_inputs():
    list_of_inputs = []

    # Input 1: Zero window size
    input_dict = {"M": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single element window
    input_dict = {"M": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two elements window
    input_dict = {"M": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small even window size
    input_dict = {"M": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small odd window size
    input_dict = {"M": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Medium even window size
    input_dict = {"M": 16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Medium odd window size
    input_dict = {"M": 25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Standard power-of-two size
    input_dict = {"M": 64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large window size
    input_dict = {"M": 128}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: numpy int32 type
    input_dict = {"M": np.int32(32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: numpy int64 type
    input_dict = {"M": np.int64(256)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Large power-of-two size
    input_dict = {"M": 1024}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.blackman"] = blackman_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.blackman' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.blackman'.")


check_valid('jax.numpy.blackman', generated_inputs['jax.numpy.blackman'], lib="jax", suffix=0)
