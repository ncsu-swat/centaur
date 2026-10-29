
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def hanning_inputs():
    list_of_inputs = []

    # Input 1: Zero window size
    input_dict = {"M": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Window size 1
    input_dict = {"M": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Window size 2
    input_dict = {"M": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Small odd window size
    input_dict = {"M": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small even window size
    input_dict = {"M": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Window size 5
    input_dict = {"M": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Window size 10
    input_dict = {"M": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Window size 16
    input_dict = {"M": 16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Odd window size
    input_dict = {"M": 51}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Power-of-two window size
    input_dict = {"M": 128}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Typical FFT window size
    input_dict = {"M": 1024}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.hanning"] = hanning_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hanning' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hanning'.")


check_valid('jax.numpy.hanning', generated_inputs['jax.numpy.hanning'], lib="jax", suffix=0)
