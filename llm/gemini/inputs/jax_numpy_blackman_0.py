
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def blackman_inputs():
    list_of_inputs = []

    # Input 1: Size 1 window
    input_dict = {"M": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Size 2 window
    input_dict = {"M": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Small window size (4)
    input_dict = {"M": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Medium window size (10)
    input_dict = {"M": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Odd window size (51)
    input_dict = {"M": 51}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger window size (128)
    input_dict = {"M": 128}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Power of two size (512)
    input_dict = {"M": 512}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large size (1024)
    input_dict = {"M": 1024}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Another common size (256)
    input_dict = {"M": 256}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Using numpy integer type (64)
    input_dict = {"M": np.int32(64)}
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
