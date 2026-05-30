
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bartlett_inputs():
    list_of_inputs = []

    # Input 1: Small positive integer
    list_of_inputs.append({"M": 5})

    # Input 2: Zero
    list_of_inputs.append({"M": 0})

    # Input 3: Another small positive integer
    list_of_inputs.append({"M": 3})

    # Input 4: Medium positive integer
    list_of_inputs.append({"M": 10})

    # Input 5: Large positive integer
    list_of_inputs.append({"M": 100})

    # Input 6: numpy.int32 type
    list_of_inputs.append({"M": np.int32(7)})

    # Input 7: numpy.int64 type
    list_of_inputs.append({"M": np.int64(20)})

    # Input 8: Smallest positive integer
    list_of_inputs.append({"M": 1})

    # Input 9: Even integer
    list_of_inputs.append({"M": 8})

    # Input 10: Very large integer
    list_of_inputs.append({"M": 1000})

    return list_of_inputs

generated_inputs["jax.numpy.bartlett"] = bartlett_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bartlett' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bartlett'.")


check_valid('jax.numpy.bartlett', generated_inputs['jax.numpy.bartlett'], lib="jax", suffix=0)
