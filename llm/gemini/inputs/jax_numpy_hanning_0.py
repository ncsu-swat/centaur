
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hanning_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"M": 0})

    # Input 2: One
    list_of_inputs.append({"M": 1})

    # Input 3: Small positive integer
    list_of_inputs.append({"M": 4})

    # Input 4: Standard size integer
    list_of_inputs.append({"M": 10})

    # Input 5: np.int32 type
    list_of_inputs.append({"M": np.int32(8)})

    # Input 6: np.int64 type
    list_of_inputs.append({"M": np.int64(16)})

    # Input 7: np.int16 type
    list_of_inputs.append({"M": np.int16(32)})

    # Input 8: Larger size integer
    list_of_inputs.append({"M": 100})

    # Input 9: Large power of 2
    list_of_inputs.append({"M": 1024})

    # Input 10: Odd positive integer
    list_of_inputs.append({"M": 5})

    # Input 11: Medium standard integer
    list_of_inputs.append({"M": 50})

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
