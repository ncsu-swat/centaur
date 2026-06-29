
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def factorial_inputs():
    list_of_inputs = []

    # Input 1: Standard python integer
    input_dict = {
        "n": 5,
        "exact": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero
    input_dict = {
        "n": 0,
        "exact": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger integer
    input_dict = {
        "n": 10,
        "exact": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: numpy int32
    input_dict = {
        "n": np.int32(6),
        "exact": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: numpy int64
    input_dict = {
        "n": np.int64(12),
        "exact": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy int32 another value
    input_dict = {
        "n": np.int32(4),
        "exact": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy int64 another value
    input_dict = {
        "n": np.int64(3),
        "exact": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small positive integer
    input_dict = {
        "n": 1,
        "exact": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Max safe integer representation (20)
    input_dict = {
        "n": 20,
        "exact": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another python integer
    input_dict = {
        "n": 8,
        "exact": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Another numpy int32
    input_dict = {
        "n": np.int32(15),
        "exact": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.factorial_2"] = factorial_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.factorial_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.factorial_2'.")


check_valid('jax.scipy.special.factorial', generated_inputs['jax.scipy.special.factorial_2'], lib="jax", suffix=2)
