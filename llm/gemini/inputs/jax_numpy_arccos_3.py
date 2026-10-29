
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arccos_inputs():
    list_of_inputs = []

    # Input 1: Zero (int)
    input_dict = {"x": int(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive boundary value (int)
    input_dict = {"x": int(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative boundary value (int)
    input_dict = {"x": int(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive value outside [-1, 1] (int)
    input_dict = {"x": int(2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative value outside [-1, 1] (int)
    input_dict = {"x": int(-2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive integer (int)
    input_dict = {"x": int(100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large negative integer (int)
    input_dict = {"x": int(-100)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: np.int32 scalar zero
    input_dict = {"x": np.int32(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: np.int32 scalar 1
    input_dict = {"x": np.int32(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: np.int64 scalar -1
    input_dict = {"x": np.int64(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: np.int16 scalar 5
    input_dict = {"x": np.int16(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: np.int8 scalar -10
    input_dict = {"x": np.int8(-10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.arccos_3"] = arccos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccos_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccos_3'.")


check_valid('jax.numpy.arccos', generated_inputs['jax.numpy.arccos_3'], lib="jax", suffix=3)
