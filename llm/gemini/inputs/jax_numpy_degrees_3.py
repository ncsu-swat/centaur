
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def degrees_inputs():
    list_of_inputs = []

    # Input 1: Zero
    input_dict = {"x": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive small integer
    input_dict = {"x": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative small integer
    input_dict = {"x": int(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Value corresponding to approximately pi radians
    input_dict = {"x": int(3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Value corresponding to approximately 2*pi radians
    input_dict = {"x": int(6)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative integer
    input_dict = {"x": int(-10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Moderate positive integer
    input_dict = {"x": int(42)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger positive integer
    input_dict = {"x": int(180)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger negative integer
    input_dict = {"x": int(-360)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Very large integer
    input_dict = {"x": int(1000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Power of two integer
    input_dict = {"x": int(256)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.degrees_3"] = degrees_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.degrees_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.degrees_3'.")


check_valid('jax.numpy.degrees', generated_inputs['jax.numpy.degrees_3'], lib="jax", suffix=3)
