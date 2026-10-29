
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def right_shift_inputs():
    list_of_inputs = []

    # Input 1: Basic shift
    input_dict = {"x1": 16, "x2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero shift
    input_dict = {"x1": 8, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero value shifted
    input_dict = {"x1": 0, "x2": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Shift resulting in 0
    input_dict = {"x1": 3, "x2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger shift amount
    input_dict = {"x1": 1024, "x2": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Odd number shift
    input_dict = {"x1": 255, "x2": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shift power of two to 1
    input_dict = {"x1": 64, "x2": 6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small shift on larger integer
    input_dict = {"x1": 100000, "x2": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Shift by 3
    input_dict = {"x1": 73, "x2": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1 shifted by 0
    input_dict = {"x1": 1, "x2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Shift by 7
    input_dict = {"x1": 512, "x2": 7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.right_shift_4"] = right_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.right_shift_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.right_shift_4'.")


check_valid('jax.numpy.right_shift', generated_inputs['jax.numpy.right_shift_4'], lib="jax", suffix=4)
