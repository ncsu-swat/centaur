
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def kaiser_inputs():
    list_of_inputs = []

    # Input 1: Minimal window size of 1 with beta=0
    list_of_inputs.append({"M": int(1), "beta": int(0)})

    # Input 2: Zero window size (empty array output)
    list_of_inputs.append({"M": int(0), "beta": int(5)})

    # Input 3: Small window size 2
    list_of_inputs.append({"M": int(2), "beta": int(1)})

    # Input 4: Small odd window size 3
    list_of_inputs.append({"M": int(3), "beta": int(3)})

    # Input 5: Standard window size 4 with moderate beta
    list_of_inputs.append({"M": int(4), "beta": int(2)})

    # Input 6: Window size 5 with beta=14 (often used in FIR filter design)
    list_of_inputs.append({"M": int(5), "beta": int(14)})

    # Input 7: Window size 8 with beta=6
    list_of_inputs.append({"M": int(8), "beta": int(6)})

    # Input 8: Window size 16 with beta=0 (rectangular window equivalent)
    list_of_inputs.append({"M": int(16), "beta": int(0)})

    # Input 9: Window size 32 with larger beta
    list_of_inputs.append({"M": int(32), "beta": int(10)})

    # Input 10: Window size 64 with beta=8
    list_of_inputs.append({"M": int(64), "beta": int(8)})

    # Input 11: Window size 128 with beta=5
    list_of_inputs.append({"M": int(128), "beta": int(5)})

    # Input 12: Window size 256 with beta=12
    list_of_inputs.append({"M": int(256), "beta": int(12)})

    return list_of_inputs

generated_inputs["jax.numpy.kaiser_3"] = kaiser_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.kaiser_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.kaiser_3'.")


check_valid('jax.numpy.kaiser', generated_inputs['jax.numpy.kaiser_3'], lib="jax", suffix=3)
