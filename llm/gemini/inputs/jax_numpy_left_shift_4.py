
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def left_shift_inputs():
    list_of_inputs = []

    # Input 1: Zero shift on zero
    input_dict = {"x": int(0), "y": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Shift positive integer by 1
    input_dict = {"x": int(1), "y": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Shift positive integer by multiple bits
    input_dict = {"x": int(2), "y": int(3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative integer shifted by 2
    input_dict = {"x": int(-1), "y": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger negative integer shifted
    input_dict = {"x": int(-16), "y": int(4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger positive integer shifted
    input_dict = {"x": int(1024), "y": int(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shift odd number
    input_dict = {"x": int(7), "y": int(10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative integer shifted by 1
    input_dict = {"x": int(-50), "y": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-zero integer shifted by 0
    input_dict = {"x": int(12345), "y": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Shift by 15 bits
    input_dict = {"x": int(3), "y": int(15)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative number shifted by 8 bits
    input_dict = {"x": int(-100), "y": int(8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Positive number shifted by 6 bits
    input_dict = {"x": int(42), "y": int(6)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.left_shift_4"] = left_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.left_shift_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.left_shift_4'.")


check_valid('jax.numpy.left_shift', generated_inputs['jax.numpy.left_shift_4'], lib="jax", suffix=4)
