
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bartlett_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"M": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: One
    input_dict = {"M": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Two
    input_dict = {"M": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three
    input_dict = {"M": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small even integer
    input_dict = {"M": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small odd integer
    input_dict = {"M": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Moderate even integer
    input_dict = {"M": 16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Moderate odd integer
    input_dict = {"M": 33}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer
    input_dict = {"M": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger power of 2
    input_dict = {"M": 256}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: NumPy int32 integer
    input_dict = {"M": int(np.int32(64))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: NumPy int64 integer
    input_dict = {"M": int(np.int64(51))}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
