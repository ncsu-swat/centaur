
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hamming_inputs():
    list_of_inputs = []

    # Input 1: Zero size window
    list_of_inputs.append({"M": 0})

    # Input 2: Window of size 1
    list_of_inputs.append({"M": 1})

    # Input 3: Window of size 2
    list_of_inputs.append({"M": 2})

    # Input 4: Window of size 3
    list_of_inputs.append({"M": 3})

    # Input 5: Window of size 4
    list_of_inputs.append({"M": 4})

    # Input 6: Window of size 5
    list_of_inputs.append({"M": 5})

    # Input 7: Window of size 10
    list_of_inputs.append({"M": 10})

    # Input 8: Size using np.int32 type
    list_of_inputs.append({"M": np.int32(15)})

    # Input 9: Size using np.int64 type
    list_of_inputs.append({"M": np.int64(25)})

    # Input 10: Window of size 100
    list_of_inputs.append({"M": 100})

    # Input 11: Larger window of size 512
    list_of_inputs.append({"M": 512})

    return list_of_inputs

generated_inputs["jax.numpy.hamming"] = hamming_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hamming' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hamming'.")


check_valid('jax.numpy.hamming', generated_inputs['jax.numpy.hamming'], lib="jax", suffix=0)
